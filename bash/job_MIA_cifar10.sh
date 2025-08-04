DATASET="cifar10"
METHOD=$1
N_MODEL=$2
SOURCE=$3
LR=$4
GPU=$5

SCREEN_NAME="${DATASET}_${METHOD}_${N_MODEL}_${SOURCE}"

screen -S $SCREEN_NAME -dm bash -c "
source ~/.bashrc
conda activate /projets/Zdehghani/torch_env
cd /projets/Zdehghani/MU_data_free
CUDA_VISIBLE_DEVICES=$GPU \
python checking_MIA.py \
    --method $METHOD \
    --model resnet18 \
    --n_model $N_MODEL \
    --source $SOURCE \
    --lr $LR \
    --dataset $DATASET
"