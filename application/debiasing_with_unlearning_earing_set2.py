import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
import os

# -------------------------
# Model definition
# -------------------------
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


# -------------------------
# Device & dataset
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = './data'
transform_test = transforms.Compose([
    transforms.Resize(178),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
test_dataset = datasets.CelebA(
    root=data_dir, split='test', transform=transform_test, download=True
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

earing_idx = 34
gender_idx = 20


# -------------------------
# Extract real embeddings (PRE-fc)
# -------------------------
def extract_real_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings, all_earing_labels, all_gender_labels = [], [], []
    with torch.no_grad():
        for inputs, full_labels in dataloader:
            inputs = inputs.to(device)
            earing_labels = full_labels[:, earing_idx].to(device)
            gender_labels = full_labels[:, gender_idx].to(device)

            features = model.resnet(inputs)  # backbone
            embeddings = features.view(features.size(0), -1)  # PRE-fc

            all_embeddings.append(embeddings)
            all_earing_labels.append(earing_labels)
            all_gender_labels.append(gender_labels)

    return (torch.cat(all_embeddings),
            torch.cat(all_earing_labels),
            torch.cat(all_gender_labels))


# -------------------------
# Evaluation on synthetic embeddings (optional)
# -------------------------
def evaluate_synthetic_embeddings(model, forget_embeddings, retain_embeddings,
                                  predicted_earing, predicted_gender,
                                  forget_mask, retain_mask):
    model.eval()
    with torch.no_grad():
        earing_logits_f = model.smiling_head(model.fc(forget_embeddings))
        predicted_earing_f = torch.round(torch.sigmoid(earing_logits_f)).squeeze()
        correct_forget = predicted_earing_f.eq(
            torch.zeros_like(predicted_earing_f)
        ).sum().item()
        accuracy_forget = (correct_forget / len(predicted_earing_f) * 100
                           if len(predicted_earing_f) > 0 else 0)

        earing_logits_r = model.smiling_head(model.fc(retain_embeddings))
        predicted_earing_r = torch.round(torch.sigmoid(earing_logits_r)).squeeze()
        correct_retain = predicted_earing_r.eq(
            predicted_earing[retain_mask].float()
        ).sum().item()
        accuracy_retain = (correct_retain / len(predicted_earing_r) * 100
                           if len(predicted_earing_r) > 0 else 0)
    print(f"Synthetic Forget Accuracy (Target: Not earing): {accuracy_forget:.2f}%")
    print(f"Synthetic Retain Accuracy (Target: Original earing): {accuracy_retain:.2f}%")
    return accuracy_forget, accuracy_retain

def evaluate_real_marginals(model, test_loader, device,
                            earing_idx=earing_idx, gender_idx=gender_idx):
    """
    Computes:
      - male_acc:   accuracy of earing classifier over all males (gender=1), ignoring earing value.
      - female_acc: accuracy over all females (gender=0), ignoring earing value.
      - earing_acc: accuracy over all samples with earing=1, ignoring gender.
      - nonearing_acc: accuracy over all samples with earing=0, ignoring gender.
    """
    model.eval()
    correct_male, total_male = 0, 0
    correct_female, total_female = 0, 0
    correct_earing, total_earing = 0, 0
    correct_nonearing, total_nonearing = 0, 0

    with torch.no_grad():
        for inputs, full_labels in test_loader:
            inputs = inputs.to(device)
            earing_labels = full_labels[:, earing_idx].to(device)   # 0/1
            gender_labels = full_labels[:, gender_idx].to(device)   # 0=female, 1=male

            earing_output, _ = model(inputs)  # [B,1]
            preds = torch.round(torch.sigmoid(earing_output)).squeeze()  # [B]

            male_mask = (gender_labels == 1)
            female_mask = (gender_labels == 0)
            earing_mask = (earing_labels == 1)
            nonearing_mask = (earing_labels == 0)

            if male_mask.any():
                correct_male += preds[male_mask].eq(earing_labels[male_mask]).sum().item()
                total_male += male_mask.sum().item()

            if female_mask.any():
                correct_female += preds[female_mask].eq(earing_labels[female_mask]).sum().item()
                total_female += female_mask.sum().item()

            if earing_mask.any():
                correct_earing += preds[earing_mask].eq(earing_labels[earing_mask]).sum().item()
                total_earing += earing_mask.sum().item()

            if nonearing_mask.any():
                correct_nonearing += preds[nonearing_mask].eq(earing_labels[nonearing_mask]).sum().item()
                total_nonearing += nonearing_mask.sum().item()

    male_acc = (correct_male / total_male * 100) if total_male > 0 else 0.0
    female_acc = (correct_female / total_female * 100) if total_female > 0 else 0.0
    earing_acc = (correct_earing / total_earing * 100) if total_earing > 0 else 0.0
    nonearing_acc = (correct_nonearing / total_nonearing * 100) if total_nonearing > 0 else 0.0

    print(f"[REAL TEST] Male (all)          acc: {male_acc:.2f}%")
    print(f"[REAL TEST] Female (all)        acc: {female_acc:.2f}%")
    print(f"[REAL TEST] Earing=1 (all)      acc: {earing_acc:.2f}%")
    print(f"[REAL TEST] Earing=0 (all)      acc: {nonearing_acc:.2f}%")

    return male_acc, female_acc, earing_acc, nonearing_acc



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


    # print(f"Male Accuracy (overall): {male_acc_overall:.2f}%")
    # print(f"Female Accuracy (overall): {female_acc_overall:.2f}%")
    # print(f"earing Accuracy (overall): {earing_acc_overall:.2f}%")
    # print(f"Non-earing Accuracy (overall): {non_earing_acc_overall:.2f}%")
        
    return (male_earing_acc, female_earing_acc,
            male_noearing_acc, female_noearing_acc,
            earing_acc_overall, non_earing_acc_overall,
            female_acc_overall, male_acc_overall)

 
def evaluate_real_embeddings(model, test_loader, device,
                             earing_idx=earing_idx, gender_idx=gender_idx):
    model.eval()
    total_forget, correct_forget, total_retain, correct_retain = 0, 0, 0, 0
    with torch.no_grad():
        for inputs, full_labels in test_loader:
            inputs = inputs.to(device)
            earing_labels = full_labels[:, earing_idx].to(device)
            gender_labels = full_labels[:, gender_idx].to(device)
            earing_output, _ = model(inputs)
            predicted_earing = torch.round(torch.sigmoid(earing_output)).squeeze()

            # forget: male & earing (1,1), retain: others
            forget_mask = (earing_labels == 1) & (gender_labels == 1)
            retain_mask = ~forget_mask

            if forget_mask.any():
                correct_forget += predicted_earing[forget_mask].eq(
                    earing_labels[forget_mask]
                ).sum().item()
                total_forget += forget_mask.sum().item()
            if retain_mask.any():
                correct_retain += predicted_earing[retain_mask].eq(
                    earing_labels[retain_mask]
                ).sum().item()
                total_retain += retain_mask.sum().item()

    forget_acc = (correct_forget / total_forget * 100) if total_forget > 0 else 0.0
    retain_acc = (correct_retain / total_retain * 100) if total_retain > 0 else 0.0
    print(f"[REAL TEST] Forget accuracy:  {forget_acc:.2f}%")
    print(f"[REAL TEST] Retain accuracy:  {retain_acc:.2f}%")
    return forget_acc, retain_acc

# -------------------------
# Helper: split by confidence into high / low for a group
# -------------------------
def split_by_confidence(conf, mask, top_frac=0.3, bottom_frac=0.3):
    """
    conf:    [B] tensor of confidence values (0–1)
    mask:    [B] bool tensor, group membership
    returns: (high_mask, low_mask) selecting top and bottom fractions within that group.
    """
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    high_mask = torch.zeros_like(mask, dtype=torch.bool)
    low_mask = torch.zeros_like(mask, dtype=torch.bool)

    if idx.numel() == 0:
        return high_mask, low_mask

    group_conf = conf[idx]
    n = idx.numel()

    k_top = int(max(1, top_frac * n)) if top_frac > 0 else 0
    k_bottom = int(max(1, bottom_frac * n)) if bottom_frac > 0 else 0
    if k_top + k_bottom > n:
        k_bottom = max(0, n - k_top)

    # sort ascending
    sorted_conf, sorted_idx = torch.sort(group_conf)  # ascending

    # lowest = first k_bottom
    if k_bottom > 0:
        low_idx = idx[sorted_idx[:k_bottom]]
        low_mask[low_idx] = True

    # highest = last k_top
    if k_top > 0:
        high_idx = idx[sorted_idx[-k_top:]]
        high_mask[high_idx] = True

    return high_mask, low_mask


# -------------------------
# Load model and extract real embeddings
# -------------------------
model = DualHeadResNet18()
model.load_state_dict(torch.load('best_dual_head_model_earing.pth'))
model = model.to(device)

real_embeddings, real_earing_labels, real_gender_labels = \
    extract_real_embeddings(model, test_loader, device)

# -------------------------
# Optimizer & scheduler
# -------------------------
optimizer = optim.SGD([
    {'params': model.fc.parameters()},
    {'params': model.smiling_head.parameters()},
    {'params': model.gender_head.parameters()}
], lr=0.01)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# -------------------------
# Training parameters
# -------------------------
num_epochs = 200
synthetic_batch_size = 1024
steps_per_epoch = 100

# Baseline evaluation
criterion = nn.BCEWithLogitsLoss()
orig_forget_acc, orig_retain_acc = evaluate_real_embeddings(model, test_loader, device)
orig_gap = abs(orig_forget_acc - orig_retain_acc)
print(f"[BASELINE] gap = {orig_gap:.2f}%")


# -------------------------
# Debiasing / unlearning loop
# -------------------------
for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")



    # Weight between retain vs forget losses
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
            # ----- Pseudo-labels -----
            earing_prob   = torch.sigmoid(earing_logits)     # [B]
            pseudo_earing = (earing_prob > 0.5).float()      # 1 = earing, 0 = not
            pseudo_gender = gender_logits.argmax(dim=1)      # 1 = male, 0 = female

            male   = (pseudo_gender == 1)
            female = (pseudo_gender == 0)
            earing = (pseudo_earing == 1)
            noear  = (pseudo_earing == 0)

            # Groups (based on the *predicted* labels)
            MS_mask = male   & earing      # male   & earing
            FS_mask = female & earing      # female & earing
            MN_mask = male   & noear       # male   & not earing
            FN_mask = female & noear       # female & not earing

            # Confidence for the *current* earing label:
            # if the model predicts 1, use p; if it predicts 0, use 1-p
            conf_earing = torch.where(pseudo_earing == 1,
                                    earing_prob,
                                    1.0 - earing_prob)

            # Hyperparams for splitting each group into high / low confidence
            top_frac_retain    = 0.3   # high-confidence part  -> retain
            bottom_frac_forget = 0.3   # low-confidence part   -> forget

            # 1) All male & earing -> RETAIN, regardless of confidence
            MS_retain_mask = MS_mask.clone()

            # 2) For FS / MN / FN:
            #    within each group:
            #       - top fraction by confidence   -> retain
            #       - bottom fraction by confidence -> forget
            FS_high, FS_low = split_by_confidence(
                conf_earing, FS_mask,
                top_frac=top_frac_retain,
                bottom_frac=bottom_frac_forget,
            )
            MN_high, MN_low = split_by_confidence(
                conf_earing, MN_mask,
                top_frac=top_frac_retain,
                bottom_frac=bottom_frac_forget,
            )
            FN_high, FN_low = split_by_confidence(
                conf_earing, FN_mask,
                top_frac=top_frac_retain,
                bottom_frac=bottom_frac_forget,
            )

            # Final masks:
            #   - forget: low-confidence FS/MN/FN
            #   - retain: male&earing (all) + high-confidence FS/MN/FN
            forget_mask = FS_low | MN_low | FN_low
            retain_mask = MS_retain_mask | FS_high | MN_high | FN_high

            # Safety: avoid overlap
            retain_mask = retain_mask & (~forget_mask)



        if retain_mask.sum() == 0 or forget_mask.sum() == 0:
            continue

        # Retain set loss: match pseudo earing label
        earing_logits_r = earing_logits[retain_mask].squeeze()
        target_earing_r = pseudo_earing[retain_mask].squeeze()
        loss_r = criterion(earing_logits_r, target_earing_r)

        # Forget set loss: push earing logit -> 0 (not earing)
        earing_logits_f = earing_logits[forget_mask]
        target_earing_f = torch.zeros_like(earing_logits_f)
        loss_f = criterion(earing_logits_f, target_earing_f)

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
    print(f"[SYN TRAIN] Retain accuracy: {syn_retain_acc:.2f}%")

    real_forget_acc, real_retain_acc = evaluate_real_embeddings(model, test_loader, device)

    # Evaluate on real test set
    (male_earing_acc, female_earing_acc, male_noearing_acc, female_noearing_acc,
    earing_acc_overall, non_earing_acc_overall,
    female_acc_overall, male_acc_overall) = evaluate_real_groups(model, test_loader, device)
    cur_gap = abs(real_forget_acc - real_retain_acc)
    print(f"[REAL TEST] Current gap: {cur_gap:.2f}% (baseline {orig_gap:.2f}%)")

    if cur_gap < orig_gap and real_retain_acc >= orig_retain_acc - 2:
        print(">>> Bias mitigated (gap reduced without killing retain accuracy).")
    else:
        print(">>> Bias not mitigated yet.")


    male_acc, female_acc, earing_acc, nonearing_acc = evaluate_real_marginals(model, test_loader, device)
