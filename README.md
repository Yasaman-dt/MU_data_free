# A Universal Source-Free Class Unlearning Framework via Synthetic Embeddings

A source-free framework for **class unlearning** in image classification with options for
- **FC-only** unlearning using *real* or *synthetic* embeddings,
- **partial-layer** unlearning (before the last conv),
- multiple unlearning **methods** and **backbones**,
- reproducible baselines (**Original**) and the **Oracle** (retrain-from-scratch) upper bound.

---

## Supported Methods & Models & Datasets

**Methods**  
`FT` (Finetuning), `NG` (Negative Gradient), `NGFTW` (Negative Gradient+), `RL` (Random Labels),
`SCAR`, `BS` (Boundary Shrink), `BE` (Boundary Expand), `SCRUB`, `DELETE`

**Backbones**  
`resnet18 (ResNet-18)`, `resnet50 (ResNet-50)`, `ViT (ViT-B-16)`, `swint (Swin-T)`

**Datasets**  
`cifar10`, `cifar100` , `TinyImageNet` (job scripts provided in `bash/`)

---

## 0) Environment & Requirements

All Python packages are listed in **`bash/requirements.txt`**.

```bash
# Activate your conda environment (adjust to your path if needed)
conda activate /projets/Zdehghani/torch_env

# Install dependencies
pip install -r bash/requirements.txt
```

---

## 1) Train “Original” (baseline) models

Trains N independently initialized models on the chosen dataset/backbone. These serve as your **baseline**.

```bash
cd MU_data_free/

# Example: 1 original ResNet-18 on CIFAR-10
CUDA_VISIBLE_DEVICES=0 python training_original.py \
  --model resnet18 \
  --dataset cifar10 \
  --run_original \
  --n_model 1
```

**Flags**
- `--model {resnet18|resnet50|ViT|swint}`
- `--dataset {cifar10|cifar100|TinyImageNet}`

---

## 2) Create dataset embeddings

Computes/stores feature embeddings used by several unlearning settings 

```bash
CUDA_VISIBLE_DEVICES=0 python create_embeddings.py
```

---

## 3) Evaluate originals → CSV

Produces a CSV with baseline metrics for later comparison.

```bash
CUDA_VISIBLE_DEVICES=0 python test_originalmodel.py
```

---

## 4) Train Oracle (retrained-from-scratch)

The **Oracle** is the upper bound that retrains from scratch as if the forget request were applied to the data directly.

```bash
CUDA_VISIBLE_DEVICES=0 python training_oracle.py \
  --model resnet18 \
  --dataset cifar10 \
  --mode CR \
  --run_rt_model
```

---

## 5) Unlearning Experiments

Make the job scripts executable once:

```bash
cd bash
chmod +x job_real_cifar10.sh job_synth_cifar10.sh job_part_real_cifar10.sh job_part_synth_cifar10.sh
chmod +x job_real_cifar100.sh job_synth_cifar100.sh job_part_real_cifar100.sh job_part_synth_cifar100.sh
chmod +x job_real_tiny.sh job_synth_tiny.sh job_part_real_tiny.sh job_part_synth_tiny.sh
```

### A) FC-only unlearning with **real** embeddings

Runs FC-only unlearning using embeddings computed from real data.

**Args:** `METHOD LR N_MODEL SAMPLES_PER_CLASS GPU EPOCHS MODEL`

```bash
# CIFAR-10, ResNet-18, 5000 samples/class, 1 model, 200 epochs on GPU 0
./job_real_cifar10.sh FT 0.01 1 5000 0 200 resnet18
```

---

### B) FC-only unlearning with **synthetic** samples (our framework)

Uses synthetic embeddings/samples (e.g., Gaussian) for FC-only unlearning.

**Args:** `METHOD LR N_MODEL SAMPLES_PER_CLASS GPU EPOCHS MODEL NOISE`

```bash
# Gaussian synthetic samples
./job_synth_cifar10.sh FT 0.01 1 5000 0 200 resnet18 gaussian
```

---

### C) Partial-layer unlearning (**before the last conv**) with **real** embeddings

**Args:** `METHOD LR N_MODEL SAMPLES_PER_CLASS GPU EPOCHS MODEL`

```bash
./job_part_real_cifar10.sh FT 0.01 1 5000 0 200 resnet18
```

---

### D) Partial-layer unlearning (**before the last conv**) with **synthetic** samples

**Args:** `METHOD LR N_MODEL SAMPLES_PER_CLASS GPU EPOCHS MODEL`

```bash
./job_part_synth_cifar10.sh FT 0.01 1 5000 0 200 resnet18
```

---

## Examples

```bash
# Train original ResNet-18 models on CIFAR-10
CUDA_VISIBLE_DEVICES=0 python training_original.py --model resnet18 --dataset cifar10 --run_original --n_model 1

# Evaluate originals → CSV
CUDA_VISIBLE_DEVICES=0 python test_originalmodel.py

# Train Oracle on CIFAR-100 (ViT)
CUDA_VISIBLE_DEVICES=0 python training_oracle.py --model resnet18 --dataset cifar10 --mode CR --run_rt_model

# Create embeddings
CUDA_VISIBLE_DEVICES=0 python create_embeddings.py

# FC-only unlearning (real embeddings) with DELETE on ResNet-18
./bash/job_real_cifar10.sh DELETE 0.01 1 5000 0 100 resnet18

# FC-only unlearning (synthetic) with DELETE on ResNet-18 + Gaussian
./bash/job_synth_cifar10.sh DELETE 0.01 1 5000 0 200 resnet18 gaussian

# Partial-layer unlearning (synthetic) with DELETE on ResNet-18
./bash/job_part_synth_cifar10.sh DELETE 0.01 1 5000 0 200 resnet18

# Partial-layer unlearning (real) with DELETE on ResNet-18
./bash/job_part_real_cifar10.sh DELETE 0.01 1 5000 0 200 resnet18
```




