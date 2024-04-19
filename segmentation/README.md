# Segmentation Task
For this task, we used a simple U-Net architecture with ~600k paramateres based on the examples in [MONAI](https://monai.io/) to the segmentation. We wanted to test whether adding the synthetic data to real data would improve the segmentation metrics.
## Running the Code
You can run the code with a command as below:
```bash
python segmentation.py --output_dir ./results/ --image_size 256 \
--train_batch_size 4 --num_train_epochs 300 --valid_batch_size 8 \
--train_frames ./train_frames/ --train_masks ./train_masks/ \
--valid_frames ./valid_frames/ --valid_masks ./valid_masks/ \
--val_interval 2 --num_classes 4 --lr 0.001
```
In this example, the folders are set based on the sample commands from the `data` section of this repository and will run the segmentation model training on real data. One can add synthetic images to the train folders to train with augmentation schemes (like Real+100%). 
