# Data Prepration
## Saving Data
In our study, we used a processed version of the [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html) dataset from Kaggle at this [link](https://www.kaggle.com/datasets/toygarr/camus-subject-based). The Kaggle dataset is a ready-to-use version of the original one, where all the data from the subjects are saved as a single h5 file. You can the download the data from [here](https://www.kaggle.com/datasets/toygarr/camus-subject-base).

After downloading the data, assuming that the `subject_based_dataset.hdf5` file is present in your current working directory, you can run the following command to save training and validation images and masks in the current directory as .png files.
```bash
python image_saver.py
```
After running the above command, the images will be generated in subfolders as:
```
+---train_frames
|       ch2_ed_frame_100.png
|       ch2_ed_frame_101.png
|	...
+---train_masks
|       ch2_ed_mask_100.png
|       ch2_ed_mask_101.png
|	...
+---valid_frames
|       ch2_ed_frame_0.png
|       ch2_ed_frame_1.png
|	...
+---valid_masks
|       ch2_ed_mask_0.png
|       ch2_ed_mask_1.png
|	...
```
## Metadata for Text+Segmentation Models
For training these models, the training code reads the data from a metadata.csv file where each row indicates the information for an image of a subject including the image path, the mask path, the text prompt used for training. You can run the `meta_data_generator.py` to generate such a metadata.
```bash
python meta_data_generator.py
```


