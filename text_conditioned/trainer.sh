#!/bin/bash
 
#SBATCH --job-name=train-text-image
#SBATCH --account=<account_name>
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                         
#SBATCH --mem=32G                  
#SBATCH --time=20:00:00             
#SBATCH --gpus-per-node=1
#SBATCH --output=output-text-image.txt         
#SBATCH --error=error-text-image.txt           
#SBATCH --mail-user=<email address>
#SBATCH --mail-type=ALL 
#SBATCH --constraint=gpu_mem_32    

module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
source <path to the virtualenv>/bin/activate  
cd $SLURM_SUBMIT_DIR
pwd
nvidia-smi

export HOME="<path to your new home>"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="./data/"
export OUTPUT_DIR="results/"


accelerate launch  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=60000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit=3 \
  --resume_from_checkpoint="latest" \


