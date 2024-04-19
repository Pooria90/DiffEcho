import os
import matplotlib.pyplot as plt
import h5py

f = h5py.File("./subject_based_dataset.hdf5", "r")

# uncomment the following line to see available keys inside the h5 file
# print (f.keys())

frames2ched = f["train 2ch ED frames"][:, :, :, :]
frames2ches = f["train 2ch ES frames"][:, :, :, :]
frames4ched = f["train 4ch ED frames"][:, :, :, :]
frames4ches = f["train 4ch ES frames"][:, :, :, :]
masks2ched = f["train 2ch ED masks"][:, :, :, :]
masks2ches = f["train 2ch ES masks"][:, :, :, :]
masks4ched = f["train 4ch ED masks"][:, :, :, :]
masks4ches = f["train 4ch ES masks"][:, :, :, :]

# uncomment the following to view the sizes of arrays
# print (frames2ched.shape,frames2ches.shape,frames4ched.shape,frames4ches.shape,masks2ched.shape,masks2ches.shape,masks4ched.shape,masks4ches.shape)

# validation images: the first 50 subjects
v_indices = list(range(0, 50))

# creating path variables and directories for training data
train_images_path = "./train_frames"
train_masks_path = "./train_masks"
if not os.path.exists(train_images_path):
    os.mkdir(train_images_path)
if not os.path.exists(train_masks_path):
    os.mkdir(train_masks_path)

# creating path variables and directories for validation data
valid_images_path = "./valid_frames"
valid_masks_path = "./valid_masks"
if not os.path.exists(valid_images_path):
    os.mkdir(valid_images_path)
if not os.path.exists(valid_masks_path):
    os.mkdir(valid_masks_path)

# saving images as grayscale .png files
for i in range(len(frames2ched)):
    # deciding whether the current index belongs to validation or training set
    if i in v_indices:
        image_path = valid_images_path
        mask_path = valid_masks_path
    else:
        image_path = train_images_path
        mask_path = train_masks_path

    # for every subject we have four images and four masks to save
    plt.imsave(
        f"./{image_path}/ch2_ed_frame_{i}.png", frames2ched[i].squeeze(), cmap="gray"
    )
    plt.imsave(
        f"./{image_path}/ch4_ed_frame_{i}.png", frames4ched[i].squeeze(), cmap="gray"
    )
    plt.imsave(
        f"./{mask_path}/ch2_ed_mask_{i}.png", masks2ched[i].squeeze(), cmap="gray"
    )
    plt.imsave(
        f"./{mask_path}/ch4_ed_mask_{i}.png", masks4ched[i].squeeze(), cmap="gray"
    )
    plt.imsave(
        f"./{image_path}/ch2_es_frame_{i}.png", frames2ches[i].squeeze(), cmap="gray"
    )
    plt.imsave(
        f"./{image_path}/ch4_es_frame_{i}.png", frames4ches[i].squeeze(), cmap="gray"
    )
    plt.imsave(
        f"./{mask_path}/ch2_es_mask_{i}.png", masks2ches[i].squeeze(), cmap="gray"
    )
    plt.imsave(
        f"./{mask_path}/ch4_es_mask_{i}.png", masks4ches[i].squeeze(), cmap="gray"
    )
