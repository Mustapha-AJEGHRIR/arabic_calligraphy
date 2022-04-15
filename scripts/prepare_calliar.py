# %%
import numpy as np
import pickle
import io
import glob
import json
import base64
from utils.vis import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML, Video
import os

# %%
dataset_path = "../Calliar/"
npy_files = glob.glob(os.path.join(dataset_path, "dataset/train/**.json"))

# Preview a single file
json_path = np.random.choice(npy_files)
drawing = json.load(open(json_path))
print(get_annotation(json_path))
data, _ = convert_3d(drawing, return_flag=True, threshold=50)
draw_strokes(data, stroke_width=8, crop=True, square=True)


# %%
# ## Character-level strokes
def draw_chars(json_path, plot=False, save_folder="../data/calliar/chars/", stroke_width=3, crop=True, square=True):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    char_drawings = generate_characters(json_path)
    for i, d in enumerate(char_drawings):
        char, drawing = list(d.items())[0]
        save_path = json_path.split("/")[-1][:-5] + f"_{i}:{char}.png"
        if save_folder and os.path.exists(os.path.join(save_folder, save_path)):
            continue
        data, _ = convert_3d(drawing, return_flag=True, threshold=50)
        im = draw_strokes(data, stroke_width=stroke_width, crop=True, square=True)
        if save_folder:
            im.save(os.path.join(save_folder, save_path))
        if plot:
            print(char)
            display(im)


draw_chars(json_path, plot=True)

# %%
from tqdm import tqdm

for json_path in tqdm(npy_files):
    draw_chars(json_path, plot=False, stroke_width=np.random.randint(2, 8))

# %%
# plot a random sample of characters
import glob
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (7, 7)
save_folder = "../data/calliar/chars/"
# plot 5x5 images from save_folder
files = glob.glob(os.path.join(save_folder, "*.png"))[:25]
fig, ax = plt.subplots(5, 5, sharex=True, sharey=True, subplot_kw={"xticks": [], "yticks": []})
ax = ax.flatten()
for i, file in enumerate(files):
    ax = plt.subplot(5, 5, i + 1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_visible(True)
        ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color("black")
    plt.imshow(plt.imread(file))
    # plt.axis("off")
    # plot label
    label = file.split("/")[-1].split(":")[-1][:-4]
    plt.title(label)
plt.tight_layout()


# %%
# ## word-level strokes
def draw_words(json_path, plot=False, save_folder="../data/calliar/words/", stroke_width=3, crop=True, square=True):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    word_drawings = generate_words(json_path)
    print(json_path)
    # print(word_drawings)
    for i, d in enumerate(word_drawings):
        word, drawing = list(d.items())[0]
        save_path = json_path.split("/")[-1][:-5] + f"_{i}:{word}.png"
        # if save_folder and os.path.exists(os.path.join(save_folder, save_path)):
        #     continue
        data, _ = convert_3d(drawing, return_flag=True, threshold=50)
        im = draw_strokes(data, stroke_width=stroke_width, crop=True, square=True)
        if save_folder:
            im.save(os.path.join(save_folder, save_path))
        if plot:
            print(word)
            display(im)


draw_words(npy_files[-8], plot=True)

# %%
from tqdm import tqdm

for json_path in tqdm(npy_files):
    draw_words(json_path, plot=False, stroke_width=np.random.randint(2, 8))
