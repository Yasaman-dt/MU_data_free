RUN_MODEL="cifar10_CR"
DATASET="cifar10"
METHOD=$1
lr=$2
N_MODEL=$3
SAMPLE_PER_CLASS=$4
gpu=$5
epoch=$6
MODEL=$7
NOISE=$8

SCREEN_NAME="SYNTH_${RUN_MODEL}_${METHOD}_${N_MODEL}_${lr}_${epoch}_${SAMPLE_PER_CLASS}_${MODEL}.sh"

screen -S $SCREEN_NAME -dm bash -c "
source ~/.bashrc
conda activate /projets/Zdehghani/torch_env
cd /projets/Zdehghani/MU_data_free
CUDA_VISIBLE_DEVICES=$gpu \
python main.py  \
    --run_name $RUN_MODEL \
    --dataset $DATASET \
    --mode CR \
    --cuda 0 \
    --save_df \
    --run_unlearn  \
    --num_workers 4 \
    --method $METHOD \
    --model $MODEL \
    --bsize 1024 \
    --lr $lr \
    --epochs $epoch  \
    --patience 50  \
    --samples_per_class $SAMPLE_PER_CLASS  \
    --n_model $N_MODEL \
    --noise_type $NOISE
"
