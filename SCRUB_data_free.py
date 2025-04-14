import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

#import wandb

#run = wandb.init()

def evaluate_embedding_accuracy(model, dataloader, device):
    """Compute accuracy on real CIFAR-10 embeddings (not images)."""
    correct, total = 0, 0
    model.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)  # Pass embeddings through model
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total if total > 0 else 0


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature = 2, logit_stand = False):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def calculate_AUS(A_test_forget, A_test_retain, Aor):
    A_test_forget = A_test_forget / 100
    A_test_retain = A_test_retain / 100
    Aor = Aor / 100
    """
    Calculate the AUS based on the given accuracy values.

    Args:
        A_test_forget (float): Accuracy on the forget test set (forgettest_val_acc).
        A_test_retain (float): Accuracy on the retain test set (retaintest_val_acc).
        Aor (float): Constant value for A_or (default is 84.72).

    Returns:
        float: The calculated AUS value.
    """
    # Calculate Delta
    delta = abs(0 - A_test_forget)
    
    # Calculate AUS
    AUS = (1 - (Aor - A_test_retain)) / (1 + delta)
    
    return AUS

def SCRUB_data_free(teacher,
                    student,
                    #retainfull_loader_train,
                    #forgetfull_loader_train,
                    retain_synth_features_train,
                    retain_synth_labels_train,
                    forget_synth_features_train,
                    forget_synth_labels_train, 
                    retainfull_loader_val,
                    forgetfull_loader_val,
                    retaintest_loader_val,
                    forgettest_loader_val,
                    Aor,
                    alpha=1.0,
                    gamma=1.0,
                    betha=1.0, 
                    lr=1e-3,
                    epochs=10,
                    temperature=1.0,
                    device='cuda',
                    patience=3,
                    batch_size=256,
                    ):
    
    # for (retainfull_features_train, retainfull_labels_train), (forgetfull_features_train, forgetfull_labels_train) in zip(retainfull_loader_train, forgetfull_loader_train):
    #     retain_features_train, retain_labels_train = retainfull_features_train.to(device), retainfull_labels_train.to(device)
    #     forget_features_train, forget_labels_train = forgetfull_features_train.to(device), forgetfull_labels_train.to(device)

    retain_synth_loader_train = DataLoader(TensorDataset(retain_synth_features_train, retain_synth_labels_train), batch_size=batch_size, shuffle=True)
    forget_synth_loader_train = DataLoader(TensorDataset(forget_synth_features_train, forget_synth_labels_train), batch_size=batch_size, shuffle=True)


    student.to(device)
    teacher.to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    loss_ce = nn.CrossEntropyLoss()
    for param in teacher.parameters():
        param.requires_grad = False
    
    for param in student.parameters():
        param.requires_grad = True
        
    for name, param in teacher.named_parameters():
        print(f"Teacher {name}: requires_grad={param.requires_grad}")

    for name, param in student.named_parameters():
        print(f"student {name}: requires_grad={param.requires_grad}")
    
    results = []
    aus_history = []  

    best_results = None
    best_aus = float('-inf')  # Maximum AUS
    best_retain_acc = float('-inf')  # Maximum retaintest_val_acc
    best_forget_acc = float('inf')  # Minimum forgettest_val_acc

    # Evaluate student model before training (Epoch 0)
    student.eval()

    retain_accuracy = evaluate_embedding_accuracy(student, retain_synth_loader_train, device)
    forget_accuracy = evaluate_embedding_accuracy(student, forget_synth_loader_train, device)

    retainfull_val_acc = evaluate_embedding_accuracy(student, retainfull_loader_val, device)
    forgetfull_val_acc = evaluate_embedding_accuracy(student, forgetfull_loader_val, device)

    retaintest_val_acc = evaluate_embedding_accuracy(student, retaintest_loader_val, device)
    forgettest_val_acc = evaluate_embedding_accuracy(student, forgettest_loader_val, device)

    AUS = calculate_AUS(forgettest_val_acc, retaintest_val_acc, Aor)

    print(f"Epoch 0 (Before Training) | "
        f"Train Retain Acc: {retain_accuracy:.2f}% | Train Forget Acc: {forget_accuracy:.2f}% | "
        f"Val Retain Full Acc: {retainfull_val_acc:.2f}% | Val Forget Full Acc: {forgetfull_val_acc:.2f}% | "
        f"Val Retain Test Acc: {retaintest_val_acc:.2f}% | Val Forget Test Acc: {forgettest_val_acc:.2f}% | "
        f"AUS: {AUS:.2f}"
    )

    # Save initial results
    results.append({
        "Epoch": 0,
        "Loss": None,  # No training yet
        "Unlearning Retain CE Loss": None,
        "Unlearning Forget CE Loss": None,
        "Unlearning Retain KD Loss": None,
        "Unlearning Forget KD Loss": None,
        "Unlearning Train Retain Acc": round(retain_accuracy, 4),
        "Unlearning Train Forget Acc": round(forget_accuracy, 4),
        "Unlearning Val Retain Full Acc": round(retainfull_val_acc, 4),
        "Unlearning Val Forget Full Acc": round(forgetfull_val_acc, 4),
        "Unlearning Val Retain Test Acc": round(retaintest_val_acc, 4),
        "Unlearning Val Forget Test Acc": round(forgettest_val_acc, 4),
        "AUS": round(AUS, 4)
    })



    for epoch in range(epochs):
        student.train()
        optimizer.zero_grad()

        retain_logits_student = student(retain_synth_features_train)
        forget_logits_student = student(forget_synth_features_train) 
        
        with torch.no_grad():
            retain_logits_teacher = teacher(retain_synth_features_train) 
            forget_logits_teacher = teacher(forget_synth_features_train)
        

        # Compute Losses
        loss_kd_retain = kd_loss(retain_logits_student, retain_logits_teacher)

        loss_kd_forget = -kd_loss(forget_logits_student, forget_logits_teacher)

        loss_ce_retain = loss_ce(retain_logits_student, retain_synth_labels_train)
        loss_ce_forget = loss_ce(forget_logits_student, forget_synth_labels_train)

        # Total loss
        loss = (alpha * loss_kd_retain) + (gamma * loss_ce_retain) + (betha * loss_kd_forget)

        # Backpropagation
        loss.backward()
        optimizer.step()
       
         
        student.eval()
        
        retain_accuracy = evaluate_embedding_accuracy(student, retain_synth_loader_train, device)
        forget_accuracy = evaluate_embedding_accuracy(student, forget_synth_loader_train, device)

        retainfull_val_acc = evaluate_embedding_accuracy(student, retainfull_loader_val, device)
        forgetfull_val_acc = evaluate_embedding_accuracy(student, forgetfull_loader_val, device)

        retaintest_val_acc = evaluate_embedding_accuracy(student, retaintest_loader_val, device)
        forgettest_val_acc = evaluate_embedding_accuracy(student, forgettest_loader_val, device)
        

        AUS = calculate_AUS(forgettest_val_acc, retaintest_val_acc, Aor)  # Calculate AUS using the accuracies

        # wandb.log({"retainfull_val_acc": retainfull_val_acc, "forgetfull_val_acc": forgetfull_val_acc,
        #            "retaintest_val_acc": retaintest_val_acc, "forgettest_val_acc": forgettest_val_acc, "AUS": AUS})

        aus_history.append(AUS)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {loss.item():.4f} | "
              f"Retain CE Loss: {loss_ce_retain.item():.4f} | Forget CE Loss: {loss_ce_forget.item():.4f} | "
              f"Retain KD Loss: {loss_kd_retain.item():.4f} | Forget KD Loss: {loss_kd_forget.item():.4f} | "
              f"Train Retain Acc: {retain_accuracy:.2f}% | Train Forget Acc: {forget_accuracy:.2f}% | "
              f"Val Retain full Acc: {retainfull_val_acc:.2f}% | Val Forget full Acc: {forgetfull_val_acc:.2f}%  | "
              f"Val Retain Test Acc: {retaintest_val_acc:.2f}% | Val Forget Test Acc: {forgettest_val_acc:.2f}% | "
              f"AUS: {AUS:.2f}"
              )

        # Update the best result
        if AUS > best_aus or retaintest_val_acc > best_retain_acc or forgettest_val_acc < best_forget_acc:
            best_aus = max(best_aus, AUS)
            best_retain_acc = max(best_retain_acc, retaintest_val_acc)
            best_forget_acc = min(best_forget_acc, forgettest_val_acc)
            
            best_results = {
                "Epoch": epoch + 1,
                "Loss": round(loss.item(), 4),
                "Unlearning Retain CE Loss": round(loss_ce_retain.item(), 4),
                "Unlearning Forget CE Loss": round(loss_ce_forget.item(), 4),
                "Unlearning Retain KD Loss": round(loss_kd_retain.item(), 4),
                "Unlearning Forget KD Loss": round(loss_kd_forget.item(), 4),
                "Unlearning Train Retain Acc": round(retain_accuracy, 4),
                "Unlearning Train Forget Acc": round(forget_accuracy, 4),
                "Unlearning Val Retain Full Acc": round(retainfull_val_acc, 4),
                "Unlearning Val Forget Full Acc": round(forgetfull_val_acc, 4),
                "Unlearning Val Retain Test Acc": round(retaintest_val_acc, 4),
                "Unlearning Val Forget Test Acc": round(forgettest_val_acc, 4),
                "AUS": round(AUS, 4)
            }
            
        if len(aus_history) > patience:
            recent_trend_aus = aus_history[-patience:]

            # Condition 1: AUS is decreasing for 'patience' epochs
            decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(len(recent_trend_aus)-1))

            # Condition 2: AUS has not changed significantly for 'patience' epochs
            no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(len(recent_trend_aus)-1))


            if decreasing_aus or no_change_aus:
                print(f"Early stopping triggered at epoch {epoch+1} due to AUS criteria.")
                break



        
        # Save results for this epoch
        results.append({
            "Epoch": epoch + 1,
            "Loss": round(loss.item(), 4),
            "Unlearning Retain CE Loss": round(loss_ce_retain.item(), 4),
            "Unlearning Forget CE Loss": round(loss_ce_forget.item(), 4),
            "Unlearning Retain KD Loss": round(loss_kd_retain.item(), 4),
            "Unlearning Forget KD Loss": round(loss_kd_forget.item(), 4),
            "Unlearning Train Retain Acc": round(retain_accuracy, 4),
            "Unlearning Train Forget Acc": round(forget_accuracy, 4),
            "Unlearning Val Retain Full Acc": round(retainfull_val_acc, 4),
            "Unlearning Val Forget Full Acc": round(forgetfull_val_acc, 4),
            "Unlearning Val Retain Test Acc": round(retaintest_val_acc, 4),
            "Unlearning Val Forget Test Acc": round(forgettest_val_acc, 4),
            "AUS": round(AUS, 4)
        })
        
    print("Training completed.")
    return best_results, results, student
