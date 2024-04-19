import os
import pandas as pd

meta_data = pd.DataFrame({"image": [], "conditioning_image": [], "text": []})
# print (meta_data)

images_path = "./train_frames/"
masks_path = "./train_masks/"

frames = sorted(os.listdir(images_path))
masks = sorted(os.listdir(masks_path))

for f, m in zip(frames, masks):
    if f[:3] == "ch2":
        view = "two-chamber"
    else:
        view = "four-chamber"
    prompt = f"an ultrasound image of heart in {view} view"

    image_path = images_path + f
    mask_path = masks_path + m
    row = pd.DataFrame(
        {"image": [image_path], "conditioning_image": [mask_path], "text": [prompt]}
    )

    meta_data = pd.concat([meta_data, row], ignore_index=True)

# print (meta_data.head(10))

meta_data.to_csv(path_or_buf="./metadata.csv", sep=",", index=False)
