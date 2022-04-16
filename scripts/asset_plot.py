#%%
import glob
import os
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# folder_path = "../assets/att_maps"
folder_path = "assets/sentences/att_maps_sentences"
# walk folder
for root, dirs, files in os.walk(folder_path):
    # print dir names
    for dir_name in tqdm(dirs):
        # plot images in dir
        dir_path = os.path.join(root, dir_name)
        imgs_paths = glob.glob(os.path.join(dir_path, "*.png"))
        imgs_paths.sort(key=lambda f: int(re.sub("\D", "", f)))
        fig, ax = plt.subplots(len(imgs_paths), 1, figsize=(5, 25), sharex=True, sharey=True)
        for i, img_path in enumerate(imgs_paths):
            img = plt.imread(img_path)
            # ax[i].imshow(img)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            # ax[i].set_title(img_path.split("/")[-1])
        # plt.tight_layout()
        save_path = os.path.join(root, f"../att_maps_2/att_maps_{dir_name}.png")
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)


# %%
