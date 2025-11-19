import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import os

# Model definition
class DualHeadResNet18(nn.Module):
    def __init__(self):
        super(DualHeadResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc_input_features = 512
        self.fc = nn.Linear(self.fc_input_features, 512)
        self.smiling_head = nn.Linear(512, 1)
        self.gender_head = nn.Linear(512, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        smiling_output = self.smiling_head(x)
        gender_output = self.gender_head(x)
        return smiling_output, gender_output


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset setup
data_dir = './data'
transform_test = transforms.Compose([
    transforms.Resize(178),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = datasets.CelebA(root=data_dir, split='test', transform=transform_test, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

earing_idx = 34
gender_idx = 20

# Extract real embeddings
def extract_real_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings, all_earing_labels, all_gender_labels = [], [], []
    with torch.no_grad():
        for inputs, full_labels in dataloader:
            inputs = inputs.to(device)
            earing_labels = full_labels[:, earing_idx].to(device)
            gender_labels = full_labels[:, gender_idx].to(device)

            features = model.resnet(inputs)                # backbone
            embeddings = features.view(features.size(0), -1)  # PRE-fc
            # embeddings shape: [B, 512], same as syn_feat

            all_embeddings.append(embeddings)
            all_earing_labels.append(earing_labels)
            all_gender_labels.append(gender_labels)

    return torch.cat(all_embeddings), torch.cat(all_earing_labels), torch.cat(all_gender_labels)


# Evaluation functions
def evaluate_synthetic_embeddings(model, forget_embeddings, retain_embeddings, predicted_earing, predicted_gender, forget_mask, retain_mask):
    model.eval()
    with torch.no_grad():
        earing_logits_f = model.smiling_head(model.fc(forget_embeddings))
        predicted_earing_f = torch.round(torch.sigmoid(earing_logits_f)).squeeze()
        correct_forget = predicted_earing_f.eq(torch.zeros_like(predicted_earing_f)).sum().item()
        accuracy_forget = correct_forget / len(predicted_earing_f) * 100 if len(predicted_earing_f) > 0 else 0
        earing_logits_r = model.smiling_head(model.fc(retain_embeddings))
        predicted_earing_r = torch.round(torch.sigmoid(earing_logits_r)).squeeze()
        correct_retain = predicted_earing_r.eq(predicted_earing[retain_mask].float()).sum().item()
        accuracy_retain = correct_retain / len(predicted_earing_r) * 100 if len(predicted_earing_r) > 0 else 0
    print(f"Synthetic Forget Accuracy (Target: Not earing): {accuracy_forget:.2f}%")
    print(f"Synthetic Retain Accuracy (Target: Original earing): {accuracy_retain:.2f}%")
    return accuracy_forget, accuracy_retain


 
def evaluate_real_embeddings(model, test_loader, device, earing_idx=earing_idx, gender_idx=gender_idx):
    model.eval()
    total_forget, correct_forget, total_retain, correct_retain = 0, 0, 0, 0
    with torch.no_grad():
        for inputs, full_labels in test_loader:
            inputs = inputs.to(device)
            earing_labels = full_labels[:, earing_idx].to(device)
            gender_labels = full_labels[:, gender_idx].to(device)
            earing_output, _ = model(inputs)
            predicted_earing = torch.round(torch.sigmoid(earing_output)).squeeze()
            retain_mask = (earing_labels == 1) & (gender_labels == 1)
            forget_mask = ~retain_mask

            if forget_mask.any():
                correct_forget += predicted_earing[forget_mask].eq(earing_labels[forget_mask]).sum().item()
                total_forget += forget_mask.sum().item()
            if retain_mask.any():
                correct_retain += predicted_earing[retain_mask].eq(earing_labels[retain_mask]).sum().item()
                total_retain += retain_mask.sum().item()
    forget_acc = (correct_forget / total_forget * 100) if total_forget > 0 else 0.0
    retain_acc = (correct_retain / total_retain * 100) if total_retain > 0 else 0.0
    print(f"[REAL TEST] Forget accuracy:  {forget_acc:.2f}%")
    print(f"[REAL TEST] Retain accuracy:       {retain_acc:.2f}%")
    return forget_acc, retain_acc


def evaluate_real_groups(model, test_loader, device, earing_idx=earing_idx, gender_idx=gender_idx):
    model.eval()
    correct_male_earing, total_male_earing = 0, 0
    correct_female_earing, total_female_earing = 0, 0
    correct_male_noearing, total_male_noearing = 0, 0
    correct_female_noearing, total_female_noearing = 0, 0
    with torch.no_grad():
        for inputs, full_labels in test_loader:
            inputs = inputs.to(device)
            earing_labels = full_labels[:, earing_idx].to(device)
            gender_labels = full_labels[:, gender_idx].to(device)
            earing_output, _ = model(inputs)
            predicted_earing = torch.round(torch.sigmoid(earing_output)).squeeze()
            male_mask = (gender_labels == 1)
            female_mask = (gender_labels == 0)
            earing_mask = (earing_labels == 1)
            noearing_mask = (earing_labels == 0)
            ms_mask = male_mask & earing_mask
            if ms_mask.any():
                correct_male_earing += predicted_earing[ms_mask].eq(earing_labels[ms_mask]).sum().item()
                total_male_earing += ms_mask.sum().item()
            fs_mask = female_mask & earing_mask
            if fs_mask.any():
                correct_female_earing += predicted_earing[fs_mask].eq(earing_labels[fs_mask]).sum().item()
                total_female_earing += fs_mask.sum().item()
            mns_mask = male_mask & noearing_mask
            if mns_mask.any():
                correct_male_noearing += predicted_earing[mns_mask].eq(earing_labels[mns_mask]).sum().item()
                total_male_noearing += mns_mask.sum().item()
            fns_mask = female_mask & noearing_mask
            if fns_mask.any():
                correct_female_noearing += predicted_earing[fns_mask].eq(earing_labels[fns_mask]).sum().item()
                total_female_noearing += fns_mask.sum().item()
    male_earing_acc = (correct_male_earing / total_male_earing * 100) if total_male_earing > 0 else 0.0
    female_earing_acc = (correct_female_earing / total_female_earing * 100) if total_female_earing > 0 else 0.0
    male_noearing_acc = (correct_male_noearing / total_male_noearing * 100) if total_male_noearing > 0 else 0.0
    female_noearing_acc = (correct_female_noearing / total_female_noearing * 100) if total_female_noearing > 0 else 0.0
    print(f"[REAL TEST] Male   & earing      acc: {male_earing_acc:.2f}%")
    print(f"[REAL TEST] Female & earing      acc: {female_earing_acc:.2f}%")
    print(f"[REAL TEST] Male   & Not earing  acc: {male_noearing_acc:.2f}%")
    print(f"[REAL TEST] Female & Not earing  acc: {female_noearing_acc:.2f}%")
    
    # Ignoring gender: earing vs non-earing overall
    earing_correct = correct_male_earing + correct_female_earing
    earing_total = total_male_earing + total_female_earing

    non_earing_correct = correct_male_noearing + correct_female_noearing
    non_earing_total = total_male_noearing + total_female_noearing

    earing_acc_overall = (earing_correct / earing_total * 100) if earing_total > 0 else 0.0
    non_earing_acc_overall = (non_earing_correct / non_earing_total * 100) if non_earing_total > 0 else 0.0

    # Ignoring earing: female vs male overall
    female_correct = correct_female_earing + correct_female_noearing
    female_total = total_female_earing + total_female_noearing

    male_correct = correct_male_earing + correct_male_noearing
    male_total = total_male_earing + total_male_noearing

    female_acc_overall = (female_correct / female_total * 100) if female_total > 0 else 0.0
    male_acc_overall = (male_correct / male_total * 100) if male_total > 0 else 0.0

    print(f"\n[Ignoring gender]")
    print(f"earing Accuracy (overall): {earing_acc_overall:.2f}%")
    print(f"Non-earing Accuracy (overall): {non_earing_acc_overall:.2f}%")

    print(f"\n[Ignoring earing]")
    print(f"Female Accuracy (overall): {female_acc_overall:.2f}%")
    print(f"Male Accuracy (overall): {male_acc_overall:.2f}%")
    
    
    return (male_earing_acc, female_earing_acc,
            male_noearing_acc, female_noearing_acc,
            earing_acc_overall, non_earing_acc_overall,
            female_acc_overall, male_acc_overall)

# Load model and data
model = DualHeadResNet18()
model.load_state_dict(torch.load('best_dual_head_model_earing.pth'))
model = model.to(device)
real_embeddings, real_earing_labels, real_gender_labels = extract_real_embeddings(model, test_loader, device)

# Define loss and optimizer
optimizer = optim.SGD([
    {'params': model.fc.parameters()},
    {'params': model.smiling_head.parameters()},
    {'params': model.gender_head.parameters()}
], lr=0.01)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training parameters
num_epochs = 200
synthetic_batch_size = 1024
steps_per_epoch = 100

# Baseline evaluation
orig_forget_acc, orig_retain_acc = evaluate_real_embeddings(model, test_loader, device)
orig_gap = abs(orig_forget_acc - orig_retain_acc)
print(f"[BASELINE] gap = {orig_gap:.2f}%")

criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")

    # Evaluate on real test set
    real_forget_acc, real_retain_acc = evaluate_real_embeddings(model, test_loader, device)
    (ms_acc, fs_acc, mns_acc, fns_acc,
    earing_acc_overall, non_earing_acc_overall,
    female_acc_overall, male_acc_overall) = evaluate_real_groups(model, test_loader, device)
    cur_gap = abs(real_forget_acc - real_retain_acc)
    print(f"[REAL TEST] Current gap: {cur_gap:.2f}% (baseline {orig_gap:.2f}%)")

    if cur_gap < orig_gap and real_retain_acc >= orig_retain_acc - 2:
        print(">>> Bias mitigated (gap reduced without killing retain accuracy).")
    else:
        print(">>> Bias not mitigated yet.")

    # Adjust beta based on current gap
    beta = 0.8

    # Train on synthetic embeddings
    model.train()
    syn_correct_forget, syn_total_forget = 0, 0
    syn_correct_retain, syn_total_retain = 0, 0

    for step in range(steps_per_epoch):
        syn_feat = torch.randn(synthetic_batch_size, 512, device=device)
        z = model.fc(syn_feat)
        earing_logits = model.smiling_head(z).squeeze(1)
        gender_logits = model.gender_head(z)

        with torch.no_grad():
            earing_prob = torch.sigmoid(earing_logits)
            pseudo_earing = (earing_prob > 0.5).float()
            pseudo_gender = gender_logits.argmax(dim=1)
            retain_mask = (pseudo_earing == 1) & (pseudo_gender == 1)
            forget_mask = ~retain_mask


        if retain_mask.sum() == 0 or forget_mask.sum() == 0:
            continue

        # Retain set
        earing_logits_r = earing_logits[retain_mask].squeeze()
        gender_logits_r = gender_logits[retain_mask]
        target_earing_r = pseudo_earing[retain_mask].squeeze()
        target_gender_r = pseudo_gender[retain_mask].long()
        loss_r = criterion(earing_logits_r, target_earing_r)

        # Forget set
        earing_logits_f = earing_logits[forget_mask]
        gender_logits_f = gender_logits[forget_mask]
        target_earing_f = torch.zeros_like(earing_logits_f)
        target_gender_f = pseudo_gender[forget_mask].long()
        loss_f = criterion(earing_logits_f, target_earing_f)

        Nr = retain_mask.sum().item()
        Nf = forget_mask.sum().item()
        retain_weight = beta 
        forget_weight = 1.0 - beta
        loss = retain_weight * loss_r - forget_weight * loss_f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_r = (torch.sigmoid(earing_logits_r) > 0.5).float()
            syn_correct_retain += pred_r.eq(target_earing_r).sum().item()
            syn_total_retain += target_earing_r.numel()
            pred_f = (torch.sigmoid(earing_logits_f) > 0.5).float()
            syn_correct_forget += pred_f.eq(target_earing_f).sum().item()
            syn_total_forget += target_earing_f.numel()

    # Update learning rate
    scheduler.step()

    # Print synthetic train accuracies
    syn_forget_acc = (syn_correct_forget / syn_total_forget * 100) if syn_total_forget > 0 else 0.0
    syn_retain_acc = (syn_correct_retain / syn_total_retain * 100) if syn_total_retain > 0 else 0.0
    print(f"[SYN TRAIN] Forget accuracy: {syn_forget_acc:.2f}%")
    print(f"[SYN TRAIN] Retain accuracy:     {syn_retain_acc:.2f}%")
    print(f"[REAL TEST SUMMARY] Forget: {real_forget_acc:.2f}% | Retain: {real_retain_acc:.2f}%")
