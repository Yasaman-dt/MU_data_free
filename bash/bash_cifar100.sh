#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24    # There are 40 CPU cores on Beluga GPU nodes
#SBATCH --mem=50G
#SBATCH --time=20:00:00
#SBATCH --account=def-hadi87
#SBATCH --mail-user=zahra.dehghani.t@gmail.com
#SBATCH --mail-type=ALL

source ~/.bash_profile
module load python/3.10.13
module load cuda cudnn
source /home/zahradt/projects/def-hadi87/zahradt/torch_env/bin/activate

RUN_MODEL="cifar100_CR"
DATASET="cifar100"
METHOD=$1
lr=$2
N_MODEL=$3
SAMPLE_PER_CLASS=$4
epoch=1000

cd /home/zahradt/projects/def-hadi87/zahradt/MU_data_free
python main.py  \
    --run_name $RUN_MODEL \
    --dataset $DATASET \
    --mode CR \
    --cuda 0 \
    --save_model \
    --save_df \
    --run_unlearn  \
    --num_workers 4 \
    --method $METHOD \
    --model resnet18 \
    --bsize 1024 \
    --lr $lr \
    --epochs $epoch  \
    --patience 20  \
    --samples_per_class $SAMPLE_PER_CLASS  \
    --n_model $N_MODEL
