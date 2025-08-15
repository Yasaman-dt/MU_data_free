# MU_data_free

conda activate /projets/Zdehghani/torch_env

cd MU_data_free/

CUDA_VISIBLE_DEVICES=0 python training_original.py --model resnet18 --dataset cifar100 --run_original --n_model 2

CUDA_VISIBLE_DEVICES=0 python test_originalmodel.py

CUDA_VISIBLE_DEVICES=0 python training_oracle.py --model resnet18 --dataset cifar10  --mode CR --run_rt_model

CUDA_VISIBLE_DEVICES=0 python  create_embeddings.py

cd bash
chmod +x job_synth_cifar10.sh 
./job_synth.sh FT 0.01 1 5000 0 200 resnet18
