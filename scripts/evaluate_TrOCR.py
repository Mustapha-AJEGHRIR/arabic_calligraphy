# %%
import pandas as pd
import glob
import os

data_path = "../data/calliar/chars/"
images = glob.glob(os.path.join(data_path, "*"))[:5000]
df = pd.DataFrame(images, columns=["file_name"])
df["text"] = df["file_name"].apply(lambda x: x.split("/")[-1].split(":")[-1][:-4])  # ":" or "\uf03a"
df

# %%
test_df = df
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
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
# processor = TrOCRProcessor.from_pretrained("./checkpoint-2000", local_files_only=True)


from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
tokenizer = processor.tokenizer
from transformers import ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
# from arabert.preprocess import ArabertPreprocessor
# arabert_prep = ArabertPreprocessor(model_name=model_name)
# from transformers import RobertaTokenizer, XLMRobertaTokenizer

# tokenizer = XLMRobertaTokenizer.from_pretrained("bhavikardeshna/xlm-roberta-base-arabic") #TODO: https://github.com/huggingface/transformers/issues/2185
processor = TrOCRProcessor(feature_extractor, tokenizer)


test_dataset = IAMDataset(root_dir=data_path, df=test_df, processor=processor)

# %%
print("Number of test examples:", len(test_dataset))

from torch.utils.data import DataLoader

test_dataloader = DataLoader(test_dataset, batch_size=16)
batch = next(iter(test_dataloader))
for k, v in batch.items():
    print(k, v.shape)

labels = batch["labels"]
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.batch_decode(labels, skip_special_tokens=True)
print(label_str)

# %%
from transformers import VisionEncoderDecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("./checkpoint-2000", local_files_only=True)
model.to(device)

# %%
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 1  # 4


# %%
from datasets import load_metric

cer = load_metric("cer")

# %%
from tqdm import tqdm

print("Running evaluation...")

label_true = []
label_pred = []
for batch in tqdm(test_dataloader):
    # predict using generate
    pixel_values = batch["pixel_values"].to(device)
    outputs = model.generate(pixel_values)

    # decode
    pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
    # print("pred_str:", pred_str)
    labels = batch["labels"]
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    label_true.extend(label_str)
    label_pred.extend(pred_str)
    # remove empty strings
    for i in range(len(label_true) - 1, -1, -1):
        if len(label_true[i]) == 0:
            label_true.pop(i)
            label_pred.pop(i)
    # add batch to metric
    cer.add_batch(predictions=pred_str, references=label_str)


# save label_true and label_pred as csv
df = pd.DataFrame({"label_true": label_true, "label_pred": label_pred})
df.to_csv("label_true_pred.csv", index=False)

final_score = cer.compute()
print("Character error rate on test set:", final_score)
# %%
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 10)
# get unique labels
labels = list(set(label_true))
print(f"labels {len(labels)}: {labels}")
ConfusionMatrixDisplay.from_predictions(label_true, label_pred, labels=labels)
plt.savefig("confusion_matrix.png")

# classification report
from sklearn.metrics import classification_report

print(classification_report(label_true, label_pred, labels=labels, target_names=labels))
# save report
with open("classification_report.txt", "w") as f:
    f.write(classification_report(label_true, label_pred, labels=labels, target_names=labels))

#%%
