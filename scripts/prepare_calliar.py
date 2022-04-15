import base64
import glob
import io
import json
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML, Video, display
import argparse

from utils.vis import *

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


def plot_sample():
    # plot a random sample of characters
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


# ## word-level strokes
def draw_words(json_path, plot=False, save_folder="../data/calliar/words/", stroke_width=3, crop=True, square=True):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    word_drawings = generate_words(json_path)
    # print(json_path)
    for i, d in enumerate(word_drawings):
        word, drawing = list(d.items())[0]
        save_path = json_path.split("/")[-1][:-5] + f"_{i}:{word}.png"
        if save_folder and os.path.exists(os.path.join(save_folder, save_path)):
            continue
        data, _ = convert_3d(drawing, return_flag=True, threshold=50)
        im = draw_strokes(data, stroke_width=stroke_width, crop=True, square=True)
        if save_folder:
            im.save(os.path.join(save_folder, save_path))
        if plot:
            print(word)
            display(im)


# ## sentence-level strokes
def draw_sentences(
    json_path, plot=False, save_folder="../data/calliar/sentences/", stroke_width=3, crop=True, square=True
):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    drawing = json.load(open(json_path))
    sentence = get_annotation(json_path)
    save_path = json_path.split("/")[-1][:-5] + ".png"
    if save_folder and os.path.exists(os.path.join(save_folder, save_path)):
        return
    data, _ = convert_3d(drawing, return_flag=True, threshold=50)
    im = draw_strokes(data, stroke_width=stroke_width, crop=True, square=True)
    if save_folder:
        im.save(os.path.join(save_folder, save_path))
    if plot:
        print(sentence)
        display(im)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../Calliar/")
    parser.add_argument("--save_folder", type=str, default="../data/calliar/")
    parser.add_argument("--level", type=str, default="sentences", help="sentences, words or chars")
    parser.add_argument("--stroke_width", type=int, default=0)
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_square", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    npy_files = glob.glob(os.path.join(args.dataset_path, "dataset/train/**.json"))
    if len(npy_files) == 0:
        raise ValueError("No json files found in {}".format(args.dataset_path))

    # Preview a single file
    # json_path = np.random.choice(npy_files)
    # drawing = json.load(open(json_path))
    # print(get_annotation(json_path))
    # data, _ = convert_3d(drawing, return_flag=True, threshold=50)
    # draw_strokes(data, stroke_width=8, crop=True, square=True)

    print(f"{len(npy_files)} files found in {os.path.join(args.dataset_path, 'dataset/train/**.json')}")
    print(f"Saving to {os.path.join(args.save_folder, args.level)}")

    if args.level == "sentences":
        for json_path in tqdm(npy_files):
            draw_sentences(
                json_path,
                plot=args.plot,
                save_folder=args.save_folder,
                stroke_width=np.random.randint(2, 8) if args.stroke_width == 0 else args.stroke_width,
                crop=not args.no_crop,
                square=not args.no_square,
            )
    elif args.level == "words":
        for json_path in tqdm(npy_files):
            draw_words(
                json_path,
                plot=args.plot,
                save_folder=args.save_folder,
                stroke_width=np.random.randint(2, 8) if args.stroke_width == 0 else args.stroke_width,
                crop=not args.no_crop,
                square=not args.no_square,
            )
    elif args.level == "chars":
        for json_path in tqdm(npy_files):
            draw_chars(
                json_path,
                plot=args.plot,
                save_folder=args.save_folder,
                stroke_width=np.random.randint(2, 8) if args.stroke_width == 0 else args.stroke_width,
                crop=not args.no_crop,
                square=not args.no_square,
            )
    else:
        raise ValueError("level must be one of sentences, words or chars")
            