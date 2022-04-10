# %% [markdown]
# ## Prepare data

# %%
import wandb

wandb.init(project="TrOCR")

# %%
import pandas as pd
import glob
import os

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/calliar/chars/")
images = glob.glob(os.path.join(data_path, "*"))
df = pd.DataFrame(images, columns=["file_name"])
df["text"] = df["file_name"].apply(lambda x: x.split("/")[-1].split(":")[-1][:-4])  # ":" or "\uf03a"
df

# %%
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=8 * 4, random_state=42)
# train_df, test_df = train_df[:], test_df[:100]
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
print(f"train_df: {train_df.shape}")
print(f"test_df: {test_df.shape}")

# %%
import torch
from torch.utils.data import Dataset
from PIL import Image


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


# %%
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
tokenizer = processor.tokenizer
from transformers import ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
# from arabert.preprocess import ArabertPreprocessor
# arabert_prep = ArabertPreprocessor(model_name=model_name)
# from transformers import RobertaTokenizer, XLMRobertaTokenizer

# tokenizer = XLMRobertaTokenizer.from_pretrained("bhavikardeshna/xlm-roberta-base-arabic")
processor = TrOCRProcessor(feature_extractor, tokenizer)
train_dataset = IAMDataset(root_dir=data_path, df=train_df, processor=processor)
eval_dataset = IAMDataset(root_dir=data_path, df=test_df, processor=processor)

# %%
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

# %%
encoding = train_dataset[0]
for k, v in encoding.items():
    print(k, v.shape)

# %%
image = Image.open(train_df["file_name"][0]).convert("RGB")
image

# %%
labels = encoding["labels"]
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print("label_str", label_str)

# %% [markdown]
# ## Train a model

# %%
from transformers import VisionEncoderDecoderModel

# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-stage1")
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224-in21k", "aubmindlab/bert-base-arabertv2"
)

# %%
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
# set decoder config to causal lm (only required in case one initializes the decoder with the weights of an encoder-only model)
# this will add the randomly initialized cross-attention layers
# model.config.decoder.is_decoder = True
# model.config.decoder.add_cross_attention = True

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 1  # 4

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,
    output_dir="./",
    logging_steps=1,
    save_steps=500,
    save_total_limit=3,
    eval_steps=5,
    num_train_epochs=500,
    report_to="wandb",
    load_best_model_at_end=True,  # must be true for early stopping
)
# %%
from datasets import load_metric

cer_metric = load_metric("cer")

# %%
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


# %%
from transformers import default_data_collator
from transformers import EarlyStoppingCallback

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
)
trainer.train()

trainer.save_model()

# %%
