# Text-Conditioned Generation Task
## Overview
This section details the fine-tuning of the model using textual prompts. We utilize the pre-trained [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) model for this purpose.

## Dataset Preparation -Texual
First, prepare a dataset suitable for this project by creating a `metadata_stable_diffusion.csv` file. To do this, follow these steps:
```bash
cd data
python meta_data_generator_stable_diffusion.py
```
* Note: you can add --do_random_string flag in this part to produce random text(that model has never see)

## Model Training
To train the Stable Diffusion model capable of generating images for all classes, use the script provided below:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="<dataset name>"
export OUTPUT_DIR="results/"

accelerate launch  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=35000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit=3 \
  --resume_from_checkpoint="latest" \
```

Additionally, you can use `trainer.sh` if you are working in the Sockeye environment.

## Data Generation - Texual
To generate synthetic dataset you can use the following script:
```bash
python data_generation.py --number_of_images 200
```
Also if you are working in the Sockeye you can use data_generation script.
