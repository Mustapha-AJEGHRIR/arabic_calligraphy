import argparse
import glob
import os

import pandas as pd
import torch
import wandb
from datasets import load_metric
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    default_data_collator,
)


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/calliar/"),
    )
    parser.add_argument("--level", type=str, default="words", help="chars or words or sentences")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--predict_with_generate", type=bool, default=True)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_eval", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--early_stopping_patience", type=int, default=200)
    args = parser.parse_args()

    # load data
    data_path = os.path.join(args.data_path, args.level)
    images = glob.glob(os.path.join(data_path, "*"))
    if len(images) == 0:
        raise ValueError("No images found in {}".format(data_path))
    df = pd.DataFrame(images, columns=["file_name"])
    if args.level == "sentences":
        df["text"] = df["file_name"].apply(lambda x: x.split("/")[-1].split("_")[0])
    else:
        df["text"] = df["file_name"].apply(lambda x: x.split("/")[-1].split(":")[-1][:-4])  # ":" or "\uf03a"

    print(df.head())

    # split data
    if args.test_size > 1:
        args.test_size = int(args.test_size)
    train_df, eval_df = train_test_split(df, test_size=args.test_size, random_state=42)
    if args.limit_train is not None:
        train_df = train_df[: args.limit_train]
    if args.limit_eval is not None:
        eval_df = eval_df[: args.limit_eval]
    train_df.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
    print(f"train_df: {train_df.shape}")
    print(f"eval_df: {eval_df.shape}")

    # load feature extractors

    # from arabert.preprocess import ArabertPreprocessor
    # arabert_prep = ArabertPreprocessor(model_name=model_name)
    # from transformers import RobertaTokenizer, XLMRobertaTokenizer
    # tokenizer = XLMRobertaTokenizer.from_pretrained("bhavikardeshna/xlm-roberta-base-arabic") #TODO: https://github.com/huggingface/transformers/issues/2185

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
    tokenizer = processor.tokenizer
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    processor = TrOCRProcessor(feature_extractor, tokenizer)

    # load dataloaders
    train_dataset = IAMDataset(root_dir=args.data_path, df=train_df, processor=processor)
    eval_dataset = IAMDataset(root_dir=args.data_path, df=eval_df, processor=processor)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    # load model
    # model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-stage1")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k", "aubmindlab/bert-base-arabertv2"
    )

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    # set decoder config to causal lm (only required in case one initializes the decoder with the weights of an encoder-only model)
    # this will add the randomly initialized cross-attention layers
    # model.config.decoder.is_decoder = True
    # model.config.decoder.add_cross_attention = True # https://github.com/huggingface/transformers/issues/14195

    # set beam search parameters (comment to use greedy search)
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = args.num_beams

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=args.predict_with_generate,
        evaluation_strategy=args.evaluation_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        report_to=args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,  # must be true for early stopping
    )

    cer_metric = load_metric("cer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    wandb.init(project="TrOCR", config=args)

    trainer.train()

    trainer.save_model()
