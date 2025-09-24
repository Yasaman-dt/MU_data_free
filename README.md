# MU_data_free

# 0) Environment & Requirements
**Activate your conda env**
conda activate torch_env
pip install -r bash/requirements.txt


# 1) Train “Original” (baseline) models 
cd MU_data_free/
**models: resnet18, resnet50, ViT, swint**
CUDA_VISIBLE_DEVICES=0 \
python training_original.py \
  --model resnet18 \
  --dataset cifar10 \
  --run_original \
  --n_model 1


# 2) Create dataset embeddings
CUDA_VISIBLE_DEVICES=0 python create_embeddings.py


# 3) Evaluate originals
CUDA_VISIBLE_DEVICES=0 python test_originalmodel.py


# 4) Train Oracle (retrained-from-scratch) model
CUDA_VISIBLE_DEVICES=0 \
python training_oracle.py \
  --model resnet18 \
  --dataset cifar10 \
  --mode CR \
  --run_rt_model


# 5) Unlearning Experiments
**methods: FT(Finetuning), NG(Negative Gradient), NGFTW(Negative Gradient+), RL(Random Labels), BS(Boundary Shrink), BE(Boundary Expanding), SCRUB, SCAR, DELETE**

cd bash

chmod +x job_real_cifar10.sh job_synth_cifar10.sh job_part_real_cifar10.sh job_part_synth_cifar10.sh

chmod +x job_real_cifar100.sh job_synth_cifar100.sh job_part_real_cifar100.sh job_part_synth_cifar100.sh

chmod +x job_real_tiny.sh job_synth_tiny.sh job_part_real_tiny.sh job_part_synth_tiny.sh


**A) FC-only unlearning with real embeddings**

#CIFAR-10, ResNet-18, 5000 samples/class, 1 model, 200 epochs on GPU 0

./job_real_cifar10.sh FT 0.01 1 5000 0 200 resnet18

**B) FC-only unlearning with synthetic samples (our framework)**

./job_synth_cifar10.sh FT 0.01 1 5000 0 200 resnet18 gaussian

**C) Unlearning before the last conv for ResNet-18 Architecture**

./job_part_real_cifar10.sh FT 0.01 1 5000 0 200 resnet18

**D) Unlearning before the last conv for ResNet-18 Architecture**

./job_part_synth_cifar10.sh FT 0.01 1 5000 0 200 resnet18



