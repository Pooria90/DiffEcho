# Text+Segmentation Generation Task
For this task, we relied on the great ControlNet training tutorial from HuggingFace at [here](https://huggingface.co/docs/diffusers/main/en/training/controlnet). I used Stable Diffusion v1-5 from [this link](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) as the base model for ControlNet. I downloaded all the content and put in a folder named `stable-diffusion-v1-5` which is used in the training script as the `MODEL_DIR`.

## Training
I added a `CustomTransform` class to the `make_train_dataset` function to apply affine trainsformations on images and segemetation masks at the same time. I used the following script in Linux to run the training. 
```bash
#!/bin/bash
export MODEL_DIR="./stable-diffusion-v1-5" # This the path were I saved my SD model.
export OUTPUT_DIR="/path/for/saving/results"
export HOME="/path/to/new/linux/home" # no need to use this variable if your home directory is writtable.

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="./path/containing/matadata.csv" \
 --resolution=512 \
 --learning_rate=5e-6 \
 --checkpointing_steps=10000 \
 --max_train_steps=120000 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=1 \
 --gradient_checkpointing \
 --use_8bit_adam
```

Our training was done on four 16GB V100 GPUs. The `gradient_checkpointing` and `use_8bit_adam` were used to reduce the required GPU memory below 16GB. The training script loads images by reading addresses and text prompts from a `metadata.csv` file. For more information on this file please refer to the `data` part of this repository.

## Inference
The `infer_controlnet.py` does the inference on a single control image (segmentation map). You need to modify the paths at the begining of the code. The `data_generation.py` also does the same work, but reads control images from a folder. So, you can generate lots of images with it for downstream tasks or finding evaluation metrics like FID and KID.
