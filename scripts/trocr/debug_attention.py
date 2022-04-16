# %%
import pandas as pd
import glob
import os


print("Loading data...")
level = "sentences"
data_path = f"../../data/calliar/{level}/"
images = glob.glob(os.path.join(data_path, "*"))
df = pd.DataFrame(images, columns=["file_name"])
if len(df) == 0:
    raise ValueError("No images found in {}".format(data_path))
if level == "sentences":
    df["text"] = df["file_name"].apply(lambda x: x.split("/")[-1].split("_")[0])
else:
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


# %%

from torch.utils.data import DataLoader

test_dataset = IAMDataset(root_dir=data_path, df=test_df, processor=processor)
print("Number of test examples:", len(test_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=1)

# %%
from transformers import VisionEncoderDecoderModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VisionEncoderDecoderModel.from_pretrained("./checkpoint-1500", local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained(
    "/usr/users/gpupro/gpu_ajeghrir/projects/arabic_calligraphy/scripts/trocr/checkpoint-500", local_files_only=True
)
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


#%%
import nopdb

model.config.output_attentions = True

# Define some functions to map the tensor back to an image
import PIL
import IPython.display as ipd
import numpy as np


def inv_normalize(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (256 - 1e-5)
    return tensor


def inv_transform(tensor, normalize=True):
    tensor = inv_normalize(tensor)
    array = tensor.detach().cpu().numpy()
    array = array.transpose(1, 2, 0).astype(np.uint8)
    return PIL.Image.fromarray(array)


def plot_attention(input, attn, folder_name):
    with torch.no_grad():
        # Loop over attention heads
        for h_idx, h_weights in enumerate(attn):
            h_weights = h_weights.mean(axis=-2)  # Average over all attention keys
            h_weights = h_weights[1:]  # Skip the [class] token
            plot_weights(input, h_weights, h_idx, folder_name)
        plot_weights(input, torch.ones_like(h_weights), h_idx + 1, folder_name)


def plot_weights(input, patch_weights, h_idx, folder_name):
    if os.path.exists(folder_name) is False:
        os.mkdir(folder_name)
    # Multiply each patch of the input image by the corresponding weight
    plot = inv_normalize(input.clone())
    for i in range(patch_weights.shape[0]):
        x = i * 16 % 224
        y = i // (224 // 16) * 16
        # plot[:, y : y + 16, x : x + 16] *= patch_weights[i]
        plot[:, y : y + 16, x : x + 16] *= patch_weights[i]
    img = inv_transform(plot, normalize=False)
    # save img
    img.save(f"{folder_name}/attention_{h_idx}.png")


# attn_call.locals["outputs"][1]  # torch.Size([1, 12, 197, 197])
generator = iter(test_dataloader)
for idx in range(10):
    if os.path.exists("att_maps") == False:
        os.mkdir("att_maps")
    batch = next(generator)
    labels = batch["labels"]
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    print("label_str", label_str)

    pixel_values = batch["pixel_values"].to(device)
    with nopdb.capture_call(model.encoder.encoder.layer[11].attention.forward) as attn_call:
        outputs = model.generate(pixel_values)

    # decode
    pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
    print("pred_str:", pred_str)

    plot_attention(pixel_values[0], attn_call.locals["outputs"][1][0], "att_maps/" + str(idx))
# [e.shape for e in attn_call.locals.values() if isinstance(e, torch.Tensor)]

# %%
