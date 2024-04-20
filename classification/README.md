# Classification Task

## Overview
This part of the repository contains a deep learning project for image classification using PyTorch. It features several pre-trained models and the ability to train on custom datasets with various image types.(true images, with or without generated data for some methods written in the paper.)

## Features
- Customizable image classification using models like ResNet, VGG, and EfficientNet.
- Configuration via YAML for easy swapping of models and datasets.
- Automated training and validation processes with metrics tracking.


## Configuration
Edit the `config_training.yaml` file to set your preferred models, datasets, and paths.

## Usage
Run the script with:
```bash
python classifier_training.py --data_dir <path to the true data> --val_dir <path to true validation data> --lr <learning rate>
```

After running this code, you will get the results in this folder as txt file both for validation and training.