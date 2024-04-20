#!/bin/bash
 
#SBATCH --job-name=generation-unconditional
#SBATCH --account=<your account>
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                       
#SBATCH --mem=32G                  
#SBATCH --time=20:00:00             
#SBATCH --gpus-per-node=1
#SBATCH --output=out-generation-unconditional.txt         
#SBATCH --error=err-generation-unconditional.txt           
#SBATCH --mail-user=<email address>
#SBATCH --mail-type=ALL    
#SBATCH --constraint=gpu_mem_32 

module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
source <path to the virtualenv>/bin/activate  
cd $SLURM_SUBMIT_DIR
pwd
nvidia-smi

export HOME="<path to your new home>"

python data_generation.py --number_of_images 800

