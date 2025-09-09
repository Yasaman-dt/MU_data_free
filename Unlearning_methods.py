import torch
import torchvision
from torch import nn 
from torch import optim
from torch.nn import functional as F
from opts import OPT as opt
import pickle
from tqdm import tqdm
from utils import accuracy
import time
from copy import deepcopy
from error_propagation import Complex
import os 
import csv
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
import time
from collections import defaultdict

n_model = opt.n_model
 
    
def AUS(a_t, a_or, a_f):
    aus=(Complex(1, 0)-(a_or-a_t))/(Complex(1, 0)+abs(a_f))
    return aus

def choose_method(name):
    if name=='FT':
        return FineTuning
    elif name=='NG':
        return NegativeGradient
    elif name=='NGFT':
        return NGFT
    elif name=='NGFTW':
        return NGFT_weighted
    elif name=='RL':
        return RandomLabels
    elif name=='SCAR':
        return SCAR
    elif name == 'newmethod':
        return newmethod
    elif name == 'BS':
        return BoundaryShrink
    elif name == 'BE':
        return BoundaryExpanding
    elif name == 'SCRUB':
        return SCRUB
    elif name == 'DUCK':
        return DUCK
    elif name == 'RE':
        return RetrainedEmbedding
    elif name == 'LAU':
        return LAU
    elif name == 'BT':
        return BadTeacher
    elif name == 'DELETE':
        return Delete
    else:
        raise ValueError(f"[choose_method] Unknown method: {name}")
    

def get_classifier(net: nn.Module) -> nn.Module:
    """Return the classification head (works for ResNet .fc and ViT .heads)."""
    if hasattr(net, "heads"):
        return net.heads
    if hasattr(net, "fc"):
        return net.fc
    raise AttributeError("Model has neither `heads` nor `fc`.")

def set_classifier(net: nn.Module, head: nn.Module) -> None:
    """Assign a new classification head back to the model."""
    if hasattr(net, "heads"):
        net.heads = head
    elif hasattr(net, "fc"):
        net.fc = head
    else:
        raise AttributeError("Model has neither `heads` nor `fc`.")
    
def freeze_all_but_fc(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in get_classifier(model).parameters():
        p.requires_grad = True
        
        
def compute_avg_fc_gradients_per_class(net, dataloader, device, num_classes=None):
    """
    Computes, for each class index 0..num_classes-1, the average L2 norm of the gradient
    on the final linear’s weight row for that class. 
    If a class never appears in `dataloader`, its avg‐grad is set to 0.0.
    """
    net.eval()

    # Find the actual nn.Linear inside net.fc (or raise if none found)
    # — this assumes there is exactly one Linear layer in net.fc.modules().
    linear_layer = None
    for m in net.modules():
        if isinstance(m, nn.Linear):
            linear_layer = m
    if linear_layer is None:
        raise RuntimeError("No nn.Linear found inside classifier head")

    # If the caller didn’t pass num_classes, infer from linear.weight shape
    if num_classes is None:
        num_classes = linear_layer.weight.shape[0]

    fc_grads = defaultdict(list)

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        net.zero_grad()

        # forward+backward
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        # grad is [num_classes × emb_dim]
        grad = linear_layer.weight.grad.detach()
        # accumulate L2‐norm of each row
        for class_name in torch.unique(targets).tolist():
            # Only measure gradient‐norm for the classes actually in this batch
            grad_norm = grad[class_name].norm().item()
            fc_grads[class_name].append(grad_norm)

    # average (or 0.0 if never seen)
    avg_grads = {}
    for class_name in range(num_classes):
        glist = fc_grads.get(class_name, [])
        if len(glist) > 0:
            avg_grads[class_name] = sum(glist) / len(glist)
        else:
            avg_grads[class_name] = 0.0

    return avg_grads



def count_samples(dataloader):
    return sum(inputs.size(0) for inputs, _ in dataloader)
        
def calculate_accuracy(net, dataloader, use_fc_only=False):
    net.eval()
    clf = get_classifier(net) if use_fc_only else None
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            if use_fc_only:
                outputs = clf(inputs)  # Only use the FC layer
            else:
                outputs = net(inputs)     # Full model
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

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

    return correct / total if total > 0 else 0



def log_epoch_to_csv(epoch, epoch_times,train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, mode, dataset, model, class_to_remove, seed, retain_count, forget_count,total_count):
    os.makedirs(f'results_synth_{opt.noise_type}/samples_per_class_{opt.samples_per_class}/{mode}/epoch_logs_m{n_model}_lr{opt.lr_unlearn}', exist_ok=True)

    if isinstance(class_to_remove, list):
        class_name = '_'.join(map(str, class_to_remove))
    else:
        class_name = class_to_remove if class_to_remove is not None else 'all'

    csv_path = f'results_synth_{opt.noise_type}/samples_per_class_{opt.samples_per_class}/{mode}/epoch_logs_m{n_model}_lr{opt.lr_unlearn}/{dataset}_{model}_epoch_results_m{n_model}_{class_name}.csv'
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'epoch_times', 'mode', 'Forget Class', 'seed', 'train_retain_acc', 'train_fgt_acc', 'val_test_retain_acc', 'val_test_fgt_acc', 'val_full_retain_acc', 'val_full_fgt_acc', 'AUS', 'retain_count', 'forget_count','total_count'])
        writer.writerow([epoch, epoch_times, mode, class_name, seed, train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, retain_count, forget_count,total_count])

def log_summary_across_classes(best_epoch, train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, mode, dataset, model, class_to_remove, seed, retain_count, forget_count,total_count, unlearning_time_until_best):
    os.makedirs(f'results_synth_{opt.noise_type}', exist_ok=True)
    summary_path = f'results_synth_{opt.noise_type}/samples_per_class_{opt.samples_per_class}/{mode}/{dataset}_{model}_unlearning_summary_m{n_model}_lr{opt.lr_unlearn}.csv'
    file_exists = os.path.isfile(summary_path)

    if isinstance(class_to_remove, list):
        class_name = '_'.join(map(str, class_to_remove))
    else:
        class_name = class_to_remove if class_to_remove is not None else 'all'

    with open(summary_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'Forget Class', 'seed', 'mode', 'dataset', 'model', 'train_retain_acc', 'train_fgt_acc', 'val_test_retain_acc', 'val_test_fgt_acc', 'val_full_retain_acc', 'val_full_fgt_acc', 'AUS', 'retain_count', 'forget_count','total_count','unlearning_time'])
        writer.writerow([best_epoch, class_name, seed, mode, dataset, model, train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, retain_count, forget_count,total_count, unlearning_time_until_best])

def calculate_AUS(A_test_forget, A_test_retain, Aor):
    A_test_forget = A_test_forget
    A_test_retain = A_test_retain
    Aor = Aor
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
    
    
        
class BaseMethod:
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real):
        self.net = net.to(opt.device)
        self.train_retain_loader = train_retain_loader
        self.train_fgt_loader = train_fgt_loader
        self.test_retain_loader = test_retain_loader
        self.test_fgt_loader = test_fgt_loader
        self.retainfull_loader_real = retainfull_loader_real
        self.forgetfull_loader_real = forgetfull_loader_real
        self.head_fc = get_classifier(self.net)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(get_classifier(self.net).parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
        self.epochs = opt.epochs_unlearn
        self.target_accuracy = opt.target_accuracy
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.scheduler, gamma=0.5)

        print(f"Train Retain Samples: {len(self.train_retain_loader.dataset)}")
        print(f"Test Retain Samples: {len(self.test_retain_loader.dataset)}")
        print(f"Real Retain Full Samples: {len(self.retainfull_loader_real.dataset)}")
        print(f"Train forget Samples: {len(self.train_fgt_loader.dataset)}")
        print(f"Test forget Samples: {len(self.test_fgt_loader.dataset)}")
        print(f"Real forget Full Samples: {len(self.forgetfull_loader_real.dataset)}")

    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        self.net.train()
        best_model_state = None
        best_aus = -float('inf')
        best_epoch = -1
        patience_counter = 0
        patience = opt.patience

        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs

        aus_history = []
        results = []
        a_or_value = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        epoch_times = []

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()

            for inputs, targets in self.loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(inputs, targets)
                loss.backward()
                self.optimizer.step()


            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)

            with torch.no_grad():
                self.net.eval()
                acc_train_ret = calculate_accuracy(self.net, self.train_retain_loader, use_fc_only=True)
                acc_train_fgt = calculate_accuracy(self.net, self.train_fgt_loader, use_fc_only=True)
                acc_test_val_ret = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
                acc_test_val_fgt = calculate_accuracy(self.net, self.test_fgt_loader, use_fc_only=True)
                acc_full_val_ret = calculate_accuracy(self.net, self.retainfull_loader_real, use_fc_only=True)
                acc_full_val_fgt = calculate_accuracy(self.net, self.forgetfull_loader_real, use_fc_only=True)

                self.net.train()
                
                a_t = Complex(acc_test_val_ret, 0.0)
                a_f = Complex(acc_test_val_fgt, 0.0)
                a_or = Complex(a_or_value, 0.0)

                aus_result = AUS(a_t, a_or, a_f)
                aus_value = aus_result.value
                aus_error = aus_result.error

                aus_history.append(aus_value)


                print(f"Train Retain Acc: {acc_train_ret:.3f},"
                      f"Train Forget Acc: {acc_train_fgt:.3f},"
                      f"Val Retain Test Acc: {acc_test_val_ret:.3f},"
                      f"Val Forget Test Acc: {acc_test_val_fgt:.3f},"
                      f"Val Retain Full Acc: {acc_full_val_ret:.3f},"
                      f"Val Forget Full Acc: {acc_full_val_fgt:.3f},"
                      f"target Acc: {self.target_accuracy:.3f},"
                      f"AUS: {aus_value:.3f}±{aus_error:.4f}")
                
                if aus_value > best_aus:
                    best_aus = aus_value
                    best_model_state = deepcopy(self.net.state_dict())
                    best_epoch = epoch

                    best_acc_train_ret = acc_train_ret
                    best_acc_train_fgt = acc_train_fgt
                    best_acc_test_val_ret = acc_test_val_ret
                    best_acc_test_val_fgt = acc_test_val_fgt
                    best_acc_full_val_ret = acc_full_val_ret
                    best_acc_full_val_fgt = acc_full_val_fgt


                    checkpoint_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                    )

                    torch.save(best_model_state, checkpoint_path)
                    print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={aus_value:.4f} to {checkpoint_path}")



                if acc_test_val_fgt == 0.0:
                    zero_acc_fgt_counter += 1
                else:
                    zero_acc_fgt_counter = 0

                if zero_acc_fgt_counter >= zero_acc_patience:
                    print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                    break
    
                if len(aus_history) > patience:
                    recent_trend_aus = aus_history[-patience:]

                    # Condition 1: AUS is decreasing
                    decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(patience - 1))

                    # Condition 2: AUS has not changed significantly
                    no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(patience - 1))

                    # Condition 3: AUS is above the minimum threshold
                    #aus_high_enough = aus_value >= 70

                    if (decreasing_aus or no_change_aus):# and aus_high_enough:
                        print(f"[Early Stopping] Triggered at epoch {epoch+1} due to AUS trend.")
                        break

                # Additional early stopping: AUS < 0.4 for more than 20 epochs
                low_aus_threshold = 0.4
                low_aus_patience = 20

                low_aus_count = sum(a < low_aus_threshold for a in aus_history[-low_aus_patience:])
                if low_aus_count >= low_aus_patience:
                    print(f"[Early Stopping] Triggered due to AUS < {low_aus_threshold} for {low_aus_patience} consecutive epochs.")
                    break

                
                ## Early stopping condition
                #if acc_test_fgt <= self.target_accuracy:
                #    patience_counter += 1
                #    if patience_counter >= patience_limit:
                #        print("[Early Stopping Triggered]")
                #        break
#
                
                # Save a summary across all unlearning runs
                log_epoch_to_csv(
                    epoch=epoch,
                    epoch_times=duration,
                    train_retain_acc=round(acc_train_ret, 4),
                    train_fgt_acc=round(acc_train_fgt, 4),
                    val_test_retain_acc=round(acc_test_val_ret, 4),
                    val_test_fgt_acc=round(acc_test_val_fgt, 4),
                    val_full_retain_acc=round(acc_full_val_ret, 4),
                    val_full_fgt_acc=round(acc_full_val_fgt, 4),
                    AUS=round(aus_value, 4),
                    mode=opt.method,
                    dataset=opt.dataset,
                    model=opt.model,
                    class_to_remove=self.class_to_remove,
                    seed=opt.seed,
                    retain_count=retain_count,
                    forget_count=forget_count,
                    total_count=total_count)

            self.scheduler.step()

        unlearning_time_until_best = sum(epoch_times[:best_epoch + 1])

        log_summary_across_classes(
            best_epoch=best_epoch,
            train_retain_acc=round(best_acc_train_ret, 4),
            train_fgt_acc=round(best_acc_train_fgt, 4),
            val_test_retain_acc=round(best_acc_test_val_ret, 4),
            val_test_fgt_acc=round(best_acc_test_val_fgt, 4),
            val_full_retain_acc=round(best_acc_full_val_ret, 4),
            val_full_fgt_acc=round(best_acc_full_val_fgt, 4),
            AUS=round(best_aus, 4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))


        self.net.eval()
        return self.net

    
class FineTuning(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        self.loader = self.train_retain_loader
        self.target_accuracy=0.0
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)

    def loss_f(self, inputs, targets, test=None):
        outputs = self.head_fc(inputs)
        loss = self.criterion(outputs, targets)
        return loss


class RandomLabels(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        self.loader = self.train_fgt_loader
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)

        if opt.mode == "CR":
            self.random_possible = torch.tensor([i for i in range(opt.num_classes) if i not in self.class_to_remove]).to(opt.device).to(torch.float32)

    def loss_f(self, inputs, targets):
        outputs = self.head_fc(inputs)
        #create a random label tensor of the same shape as the outputs chosing values from self.possible_labels
        random_labels = self.random_possible[torch.randint(low=0, high=self.random_possible.shape[0], size=targets.shape)].to(torch.int64).to(opt.device)
        loss = self.criterion(outputs, random_labels)
        return loss


class NegativeGradient(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        self.loader = self.train_fgt_loader
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)

    def loss_f(self, inputs, targets):
        outputs = self.head_fc(inputs)
        loss = self.criterion(outputs, targets) * (-1)
        return loss


class NGFT(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)

        self.train_retain_loader = train_retain_loader
        self.train_fgt_loader = train_fgt_loader
        self.test_retain_loader = test_retain_loader
        self.test_fgt_loader = test_fgt_loader
        self.retainfull_loader_real = retainfull_loader_real
        self.forgetfull_loader_real = forgetfull_loader_real
        self.class_to_remove = class_to_remove
        self.heaf_fc = get_classifier(self.net)

    def loss_r(self, inputs, targets):
        return self.criterion(self.heaf_fc(inputs), targets)
    def loss_f(self, inputs, targets):
        return -self.criterion(self.heaf_fc(inputs), targets)
        
    def run(self):
        self.net.train()
        best_model_state = None
        best_aus = -float('inf')
        best_epoch = -1
        patience_counter = 0
        patience = opt.patience

        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs

        aus_history = []
        results = []
        a_or_value = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        

        forget_loader = cycle(self.train_fgt_loader)  # repeat forever
        epoch_times = []

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()

            for inputs_r, targets_r in self.train_retain_loader:
                inputs_r, targets_r = inputs_r.to(opt.device), targets_r.to(opt.device)

                # fetch forget batch
                inputs_f, targets_f = next(forget_loader)
                inputs_f, targets_f = inputs_f.to(opt.device), targets_f.to(opt.device)

                # compute both losses
                loss_retain = self.loss_r(inputs_r, targets_r)
                loss_forget = self.loss_f(inputs_f, targets_f)

                # total loss
                total_loss = loss_retain + loss_forget

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)
            
    
            with torch.no_grad():
                self.net.eval()
                acc_train_ret = calculate_accuracy(self.net, self.train_retain_loader, use_fc_only=True)
                acc_train_fgt = calculate_accuracy(self.net, self.train_fgt_loader, use_fc_only=True)
                acc_test_val_ret = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
                acc_test_val_fgt = calculate_accuracy(self.net, self.test_fgt_loader, use_fc_only=True)
                acc_full_val_ret = calculate_accuracy(self.net, self.retainfull_loader_real, use_fc_only=True)
                acc_full_val_fgt = calculate_accuracy(self.net, self.forgetfull_loader_real, use_fc_only=True)



                self.net.train()
                
                a_t = Complex(acc_test_val_ret, 0.0)
                a_f = Complex(acc_test_val_fgt, 0.0)
                a_or = Complex(a_or_value, 0.0)

                aus_result = AUS(a_t, a_or, a_f)
                aus_value = aus_result.value
                aus_error = aus_result.error

                aus_history.append(aus_value)


                print(f"Train Retain Acc: {acc_train_ret:.3f},"
                      f"Train Forget Acc: {acc_train_fgt:.3f},"
                      f"Val Retain Test Acc: {acc_test_val_ret:.3f},"
                      f"Val Forget Test Acc: {acc_test_val_fgt:.3f},"
                      f"Val Retain Full Acc: {acc_full_val_ret:.3f},"
                      f"Val Forget Full Acc: {acc_full_val_fgt:.3f},"
                      f"target Acc: {self.target_accuracy:.3f},"
                      f"AUS: {aus_value:.3f}±{aus_error:.4f}")
                
                if aus_value > best_aus:
                    best_aus = aus_value
                    best_model_state = deepcopy(self.net.state_dict())
                    best_epoch = epoch

                    best_acc_train_ret = acc_train_ret
                    best_acc_train_fgt = acc_train_fgt
                    best_acc_test_val_ret = acc_test_val_ret
                    best_acc_test_val_fgt = acc_test_val_fgt
                    best_acc_full_val_ret = acc_full_val_ret
                    best_acc_full_val_fgt = acc_full_val_fgt
                    

                if acc_test_val_fgt == 0.0:
                    zero_acc_fgt_counter += 1
                else:
                    zero_acc_fgt_counter = 0

                if zero_acc_fgt_counter >= zero_acc_patience:
                    print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                    break
    
                if len(aus_history) > patience:
                    recent_trend_aus = aus_history[-patience:]

                    # Condition 1: AUS is decreasing
                    decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(patience - 1))

                    # Condition 2: AUS has not changed significantly
                    no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(patience - 1))

                    # Condition 3: AUS is above the minimum threshold
                    #aus_high_enough = aus_value >= 70

                    if (decreasing_aus or no_change_aus):# and aus_high_enough:
                        print(f"[Early Stopping] Triggered at epoch {epoch+1} due to AUS trend.")
                        break

                # Additional early stopping: AUS < 0.4 for more than 20 epochs
                low_aus_threshold = 0.4
                low_aus_patience = 20

                low_aus_count = sum(a < low_aus_threshold for a in aus_history[-low_aus_patience:])
                if low_aus_count >= low_aus_patience:
                    print(f"[Early Stopping] Triggered due to AUS < {low_aus_threshold} for {low_aus_patience} consecutive epochs.")
                    break

                
                ## Early stopping condition
                #if acc_test_fgt <= self.target_accuracy:
                #    patience_counter += 1
                #    if patience_counter >= patience_limit:
                #        print("[Early Stopping Triggered]")
                #        break
#
                
                # Save a summary across all unlearning runs
                log_epoch_to_csv(
                    epoch=epoch,
                    epoch_times=duration,
                    train_retain_acc=round(acc_train_ret, 4),
                    train_fgt_acc=round(acc_train_fgt, 4),
                    val_test_retain_acc=round(acc_test_val_ret, 4),
                    val_test_fgt_acc=round(acc_test_val_fgt, 4),
                    val_full_retain_acc=round(acc_full_val_ret, 4),
                    val_full_fgt_acc=round(acc_full_val_fgt, 4),
                    AUS=round(aus_value, 4),
                    mode=opt.method,
                    dataset=opt.dataset,
                    model=opt.model,
                    class_to_remove=self.class_to_remove,
                    seed=opt.seed,
                    retain_count=retain_count,
                    forget_count=forget_count,
                    total_count=total_count)

            self.scheduler.step()

        unlearning_time_until_best = sum(epoch_times[:best_epoch + 1])

        log_summary_across_classes(
            best_epoch=best_epoch,
            train_retain_acc=round(best_acc_train_ret, 4),
            train_fgt_acc=round(best_acc_train_fgt, 4),
            val_test_retain_acc=round(best_acc_test_val_ret, 4),
            val_test_fgt_acc=round(best_acc_test_val_fgt, 4),
            val_full_retain_acc=round(best_acc_full_val_ret, 4),
            val_full_fgt_acc=round(best_acc_full_val_fgt, 4),
            AUS=round(best_aus, 4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))


        self.net.eval()
        return self.net
    

class NGFT_weighted(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        
        self.train_retain_loader = train_retain_loader
        self.train_fgt_loader = train_fgt_loader
        self.test_retain_loader = test_retain_loader
        self.test_fgt_loader = test_fgt_loader
        self.retainfull_loader_real = retainfull_loader_real
        self.forgetfull_loader_real = forgetfull_loader_real
        self.class_to_remove = class_to_remove
        self.beta = 0.9  # balance factor
        self.head_fc = get_classifier(self.net)


    def loss_weighted(self, inputs_r, targets_r, inputs_f, targets_f):
        retain_loss = self.criterion(self.head_fc(inputs_r), targets_r)
        forget_loss = self.criterion(self.head_fc(inputs_f), targets_f)
        # Weighted sum as per the formula
        return self.beta * retain_loss - (1 - self.beta) * forget_loss

    def run(self):
        self.head_fc.train()
        best_model_state = None
        best_aus = -float('inf')
        best_epoch = -1
        aus_history = []
        zero_acc_fgt_counter = 0
        zero_acc_patience = 1000
        patience = opt.patience
        a_or_value = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
        forget_loader = cycle(self.train_fgt_loader)

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count

        epoch_times = []

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            for inputs_r, targets_r in self.train_retain_loader:
                inputs_r, targets_r = inputs_r.to(opt.device), targets_r.to(opt.device)
                inputs_f, targets_f = next(forget_loader)
                inputs_f, targets_f = inputs_f.to(opt.device), targets_f.to(opt.device)

                total_loss = self.loss_weighted(inputs_r, targets_r, inputs_f, targets_f)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)


            if epoch % 10 == 0:  # e.g., log every 10 epochs
                print(f"--- Gradient norms at epoch {epoch} ---")
                avg_grads_fgt = compute_avg_fc_gradients_per_class(self.head_fc, self.train_fgt_loader, device=opt.device)
                avg_grads_ret = compute_avg_fc_gradients_per_class(self.head_fc, self.train_retain_loader, device=opt.device)

                forget_classes = set(self.class_to_remove) if isinstance(self.class_to_remove, list) else {self.class_to_remove}
                for class_name in sorted(avg_grads_fgt.keys()):
                    status = "FORGET" if class_name in forget_classes else "RETAIN"
                    print(f" Class {class_name:2d} [{status}]: "
                        f"Fgt Grad = {avg_grads_fgt[class_name]:.4f}, "
                        f"Ret Grad = {avg_grads_ret.get(class_name, 0):.4f}")


            with torch.no_grad():
                self.net.eval()
                acc_train_ret = calculate_accuracy(self.net, self.train_retain_loader, use_fc_only=True)
                acc_train_fgt = calculate_accuracy(self.net, self.train_fgt_loader, use_fc_only=True)
                acc_test_val_ret = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
                acc_test_val_fgt = calculate_accuracy(self.net, self.test_fgt_loader, use_fc_only=True)
                acc_full_val_ret = calculate_accuracy(self.net, self.retainfull_loader_real, use_fc_only=True)
                acc_full_val_fgt = calculate_accuracy(self.net, self.forgetfull_loader_real, use_fc_only=True)

                self.net.train()

                a_t = Complex(acc_test_val_ret, 0.0)
                a_f = Complex(acc_test_val_fgt, 0.0)
                a_or = Complex(a_or_value, 0.0)
                aus_result = AUS(a_t, a_or, a_f)
                aus_value = aus_result.value
                aus_error = aus_result.error
                aus_history.append(aus_value)

                print(f"Train Retain Acc: {acc_train_ret:.3f},"
                      f"Train Forget Acc: {acc_train_fgt:.3f},"
                      f"Val Retain Test Acc: {acc_test_val_ret:.3f},"
                      f"Val Forget Test Acc: {acc_test_val_fgt:.3f},"
                      f"Val Retain Full Acc: {acc_full_val_ret:.3f},"
                      f"Val Forget Full Acc: {acc_full_val_fgt:.3f},"
                      f"target Acc: {self.target_accuracy:.3f},"
                      f"AUS: {aus_value:.3f}±{aus_error:.4f}")

                if aus_value > best_aus:
                    best_aus = aus_value
                    best_model_state = deepcopy(self.net.state_dict())
                    best_epoch = epoch
                    best_acc_train_ret = acc_train_ret
                    best_acc_train_fgt = acc_train_fgt
                    best_acc_test_val_ret = acc_test_val_ret
                    best_acc_test_val_fgt = acc_test_val_fgt
                    best_acc_full_val_ret = acc_full_val_ret
                    best_acc_full_val_fgt = acc_full_val_fgt

                    checkpoint_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                    )

                    torch.save(best_model_state, checkpoint_path)
                    print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={aus_value:.4f} to {checkpoint_path}")


                if acc_test_val_fgt == 0.0:
                    zero_acc_fgt_counter += 1
                else:
                    zero_acc_fgt_counter = 0

                if zero_acc_fgt_counter >= zero_acc_patience:
                    print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                    break

                if len(aus_history) > patience:
                    recent = aus_history[-patience:]
                    decreasing = all(recent[i] > recent[i+1] for i in range(patience-1))
                    no_change = all(abs(recent[i] - recent[i+1]) < 1e-4 for i in range(patience-1))
                    if decreasing or no_change:
                        print(f"[Early Stopping] Triggered at epoch {epoch+1} due to AUS trend.")
                        break

                low_aus_count = sum(a < 0.4 for a in aus_history[-20:])
                if low_aus_count >= 20:
                    print(f"[Early Stopping] Triggered due to AUS < 0.4 for 20 consecutive epochs.")
                    break

                log_epoch_to_csv(
                    epoch=epoch,
                    epoch_times=duration,
                    train_retain_acc=round(acc_train_ret, 4),
                    train_fgt_acc=round(acc_train_fgt, 4),
                    val_test_retain_acc=round(acc_test_val_ret, 4),
                    val_test_fgt_acc=round(acc_test_val_fgt, 4),
                    val_full_retain_acc=round(acc_full_val_ret, 4),
                    val_full_fgt_acc=round(acc_full_val_fgt, 4),
                    AUS=round(aus_value, 4),
                    mode=opt.method,
                    dataset=opt.dataset,
                    model=opt.model,
                    class_to_remove=self.class_to_remove,
                    seed=opt.seed,
                    retain_count=retain_count,
                    forget_count=forget_count,
                    total_count=total_count)

            self.scheduler.step()

        unlearning_time_until_best = sum(epoch_times[:best_epoch + 1])

        log_summary_across_classes(
            best_epoch=best_epoch,
            train_retain_acc=round(best_acc_train_ret, 4),
            train_fgt_acc=round(best_acc_train_fgt, 4),
            val_test_retain_acc=round(best_acc_test_val_ret, 4),
            val_test_fgt_acc=round(best_acc_test_val_fgt, 4),
            val_full_retain_acc=round(best_acc_full_val_ret, 4),
            val_full_fgt_acc=round(best_acc_full_val_fgt, 4),
            AUS=round(best_aus, 4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))


        self.net.eval()
        return self.net


class SCAR(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)
      
    def cov_mat_shrinkage(self,cov_mat,gamma1=opt.gamma1,gamma2=opt.gamma2):
        I = torch.eye(cov_mat.shape[0]).to(opt.device)
        V1 = torch.mean(torch.diagonal(cov_mat))
        off_diag = cov_mat.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        V2 = (off_diag*mask).sum() / mask.sum()
        cov_mat_shrinked = cov_mat + gamma1*I*V1 + gamma2*(1-I)*V2
        #epsilon = 1e-5
        #cov_mat_shrinked += epsilon * I  

        return cov_mat_shrinked
    
    def normalize_cov(self,cov_mat):
        sigma = torch.sqrt(torch.diagonal(cov_mat))  # standard deviations of the variables
        cov_mat = cov_mat/(torch.matmul(sigma.unsqueeze(1),sigma.unsqueeze(0)))
        return cov_mat


    def mahalanobis_dist(self, samples,samples_lab, mean,S_inv):
        #mean = mean + 1e-8
        #print(mean)
        #print(mean.shape)
        #print(F.normalize(self.tuckey_transf(samples), p=2, dim=-1)[:,None,:])
        #print(F.normalize(self.tuckey_transf(samples), p=2, dim=-1)[:,None,:].shape)
        #print(F.normalize(mean, p=2, dim=-1))
        #print(F.normalize(mean, p=2, dim=-1).shape)
        #print(samples.shape)

        #check optimized version
        diff = F.normalize(self.tuckey_transf(samples), p=2, dim=-1)[:,None,:] - F.normalize(mean, p=2, dim=-1)
        
        right_term = torch.matmul(diff.permute(1,0,2), S_inv)
        mahalanobis = torch.diagonal(torch.matmul(right_term, diff.permute(1,2,0)),dim1=1,dim2=2)
        return mahalanobis

    def distill(self, outputs_ret, outputs_original):

        soft_log_old = torch.nn.functional.log_softmax(outputs_original+10e-5, dim=1)
        soft_log_new = torch.nn.functional.log_softmax(outputs_ret+10e-5, dim=1)
        kl_div = torch.nn.functional.kl_div(soft_log_new+10e-5, soft_log_old+10e-5, reduction='batchmean', log_target=True)

        return kl_div

    #def tuckey_transf(self,vectors,delta=opt.delta):
    #    return torch.pow(vectors,delta)
    
    def tuckey_transf(self, vectors, delta=opt.delta):
        return torch.sign(vectors) * torch.pow(torch.abs(vectors), delta)

    def pairwise_cos_dist(self, x, y):
        """Compute pairwise cosine distance between two tensors"""
        x_norm = torch.norm(x, dim=1).unsqueeze(1)
        y_norm = torch.norm(y, dim=1).unsqueeze(1)
        x = x / x_norm
        y = y / y_norm
        return 1 - torch.mm(x, y.transpose(0, 1))
    
    def L2(self,embs_fgt,mu_distribs):
        embs_fgt = embs_fgt.unsqueeze(1)
        mu_distribs = mu_distribs.unsqueeze(0)
        dists=torch.norm((embs_fgt-mu_distribs),dim=2)
        return dists
 
    def run(self):
               
        """compute embeddings"""
        #if opt.model!='ViT':
        #    bbone = torch.nn.Sequential(*(list(self.net.children())[:-1] + [nn.Flatten()]))
        #    if opt.model == 'AllCNN':
        #        fc = self.net.classifier
        #    else:
        #        fc = self.net.fc
        #else:
        self.net.eval()
        fc_layer = self.net.fc

        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.head_fc.parameters():
            param.requires_grad = True
            
        original_fc = deepcopy(fc_layer) # self.net
        original_fc.eval()
 
        # embeddings of retain set
        with torch.no_grad():
             ret_embs=[]
             labs=[]
             cnt=0
             for emb_ret, lab_ret in self.train_retain_loader:
                 emb_ret, lab_ret = emb_ret.to(opt.device), lab_ret.to(opt.device)
               
                 ret_embs.append(emb_ret)
                 labs.append(lab_ret)
                 cnt+=1
             ret_embs=torch.cat(ret_embs)
             labs=torch.cat(labs)
        

            #print(ret_embs.shape)
            #print(labs.shape)
            
        # compute distribs from embeddings
        distribs=[]
        cov_matrix_inv =[]
        for i in range(opt.num_classes):
            if type(self.class_to_remove) is list:
                if i not in self.class_to_remove:
                    #print(f"Class {i} samples shape:", ret_embs[labs==i].shape)
                    samples = self.tuckey_transf(ret_embs[labs==i])
                    
                    ## DEBUG: Check for NaN or Inf
                    #print(f"Class {i} samples: mean={samples.mean().item():.6f}, std={samples.std().item():.6f}, min={samples.min().item():.6f}, max={samples.max().item():.6f}")
                    #if torch.isnan(samples).any() or torch.isinf(samples).any():
                    #    print(f"Class {i} has NaN or Inf in samples!")
                    
                    distribs.append(samples.mean(0))
                    cov = torch.cov(samples.T)
                    cov_shrinked = self.cov_mat_shrinkage(self.cov_mat_shrinkage(cov))
                    cov_shrinked = self.normalize_cov(cov_shrinked)
                    cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))

                    
                    
                #    # DEBUG: Compute eigenvalues (singular values)
                #    try:
                #        s = torch.linalg.svdvals(cov_shrinked)
                #        s_sorted, _ = torch.sort(s, descending=True)
                #        print(f"Class {i} SVD: Largest={s_sorted[0].item():.6f}, Smallest={s_sorted[-1].item():.6f}")
                #    except Exception as e:
                #        print(f"Failed to compute SVD for class {i}: {e}")
#
                #    # Try pseudo-inverse
                #    try:
                #        cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))
                #    except Exception as e:
                #        print(f"Failed to invert covariance matrix for class {i}: {e}")
                #        # Optionally: append identity matrix instead to not crash
                #        cov_matrix_inv.append(torch.eye(cov_shrinked.shape[0], device=cov_shrinked.device))
    
            else:
                print(f"Class {i} samples shape:", ret_embs[labs==i].shape)
                samples = self.tuckey_transf(ret_embs[labs==i])
                distribs.append(samples.mean(0))
                cov = torch.cov(samples.T)
                cov_shrinked = self.cov_mat_shrinkage(self.cov_mat_shrinkage(cov))
                cov_shrinked = self.normalize_cov(cov_shrinked)
                cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))


                ## DEBUG: Compute eigenvalues (singular values)
                #try:
                #    s = torch.linalg.svdvals(cov_shrinked)
                #    s_sorted, _ = torch.sort(s, descending=True)
                #    print(f"Class {i} SVD: Largest={s_sorted[0].item():.6f}, Smallest={s_sorted[-1].item():.6f}")
                #except Exception as e:
                #    print(f"Failed to compute SVD for class {i}: {e}")
#
                ## Try pseudo-inverse
                #try:
                #    cov_matrix_inv.append(torch.linalg.pinv(cov_shrinked))
                #except Exception as e:
                #    print(f"Failed to invert covariance matrix for class {i}: {e}")
                #    # Optionally: append identity matrix instead to not crash
                #    cov_matrix_inv.append(torch.eye(cov_shrinked.shape[0], device=cov_shrinked.device))
                    
        distribs=torch.stack(distribs)
        #print('distribs.shape',distribs.shape)
        cov_matrix_inv=torch.stack(cov_matrix_inv)
        
        fc_layer.train()

        optimizer = optim.Adam(fc_layer.parameters(), lr=opt.lr_unlearn, weight_decay=opt.wd_unlearn)

        init = True
        flag_exit = False
        all_closest_class = []
       
        vec_forg=None
        if 'TinyImageNet' in opt.dataset:
            th = .4
            
        else:
            th = .8

        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs
        
        aus_history = []  

        best_results = None
        best_aus = float('-inf')  # Maximum AUS
        best_retain_acc = float('-inf')  # Maximum retaintest_val_acc
        best_forget_acc = float('inf')  # Minimum forgettest_val_acc

        Aor = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
        #print('Num batch forget: ',len(self.train_fgt_loader), 'Num batch retain: ',len(self.synthetic_retain_emb_loader))

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        epoch_times = []

        for epoch in tqdm(range(opt.epochs_unlearn)):
            start_time = time.time()

            for n_batch, (embs_fgt, lab_fgt) in enumerate(self.train_fgt_loader):
                for n_batch_ret, all_batch in enumerate(self.train_retain_loader):

                    if opt.mode == 'CR':
                        embs_ret, lab_ret = all_batch
                    
                    embs_ret, lab_ret,embs_fgt, lab_fgt  = embs_ret.to(opt.device), lab_ret.to(opt.device),embs_fgt.to(opt.device), lab_fgt.to(opt.device)
                    optimizer.zero_grad()

                    # compute Mahalanobis distance between embeddings and cluster
                    dists = self.mahalanobis_dist(embs_fgt,lab_fgt,distribs,cov_matrix_inv).T  

                    if init and n_batch_ret==0:
                        closest_class = torch.argsort(dists, dim=1)
                        tmp = closest_class[:, 0]
                        closest_class = torch.where(tmp == lab_fgt, closest_class[:, 1], tmp)
                        all_closest_class.append(closest_class)
                        closest_class = all_closest_class[-1]
                    else:
                        closest_class = all_closest_class[n_batch]

                    dists = dists[torch.arange(dists.shape[0]), closest_class[:dists.shape[0]]]



                    loss_fgt = torch.mean(dists) * opt.lambda_1
                    
                    # if opt.model=='ViT':
                    #     outputs_ret = fc(bbone.forward_encoder(img_ret))
                    # else:
                    #     outputs_ret = fc(bbone(img_ret))
                    if opt.model == 'ViT':
                        outputs_ret = fc_layer(ret_embs)  
                    else:
                        outputs_ret = fc_layer(ret_embs)  
                    

                    if opt.mode =='CR':
                        with torch.no_grad():
                            #outputs_original = original_model(img_ret)
                            if opt.model == 'ViT':
                                outputs_original = original_fc(ret_embs)  
                            else:
                                outputs_original = original_fc(ret_embs) 

                            label_out = torch.argmax(outputs_original,dim=1)
                            outputs_original = outputs_original[label_out!=self.class_to_remove[0],:]
                            outputs_original[:,torch.tensor(self.class_to_remove,dtype=torch.int64)] = torch.min(outputs_original)
                        
                        outputs_ret = outputs_ret[label_out!=self.class_to_remove[0],:]
                    
                    loss_ret = self.distill(outputs_ret, outputs_original/opt.temperature)*opt.lambda_2
                    loss=loss_ret+loss_fgt
                    
                    if n_batch_ret>opt.num_retain_samp:
                        del loss,loss_ret,loss_fgt, embs_fgt,dists
                        break
                    
                    #print(f'n_batch_ret:{n_batch_ret} ,loss FGT:{loss_fgt}, loss RET:{loss_ret}')
                    loss.backward()
                    optimizer.step()

            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)
    
#                    with torch.no_grad():
#                        self.net.eval()
#                        if opt.mode=='CR':
#                            curr_acc = calculate_accuracy(self.net, self.test_fgt_loader, use_fc_only=True)
#                        self.net.train()
#                        if curr_acc <= opt.target_accuracy and epoch>1:
#                            flag_exit = True
#
#                    if flag_exit:
#                        break
#                if flag_exit:
#                    break

            # evaluate accuracy on forget set every batch
            #with torch.no_grad():
            #    self.net.eval()
            #    curr_acc = calculate_accuracy(self.net, self.test_fgt_loader, use_fc_only=True)
            #    #test_acc=calculate_accuracy(self.net, self.test)
            #    self.net.train()
            #    print(f"AAcc forget: {curr_acc:.3f}, target is {opt.target_accuracy:.3f}")
            #    if curr_acc <= opt.target_accuracy and epoch>1:
            #        flag_exit = True
#
            #if flag_exit:
            #    break
#
            init = False
            #scheduler.step()

            retain_accuracy = evaluate_embedding_accuracy(self.net.fc, self.train_retain_loader, opt.device)
            forget_accuracy = evaluate_embedding_accuracy(self.net.fc, self.train_fgt_loader, opt.device)

            retainfull_val_acc = evaluate_embedding_accuracy(self.net.fc, self.retainfull_loader_real, opt.device)
            forgetfull_val_acc = evaluate_embedding_accuracy(self.net.fc, self.forgetfull_loader_real, opt.device)

            retaintest_val_acc = evaluate_embedding_accuracy(self.net.fc, self.test_retain_loader, opt.device)
            forgettest_val_acc = evaluate_embedding_accuracy(self.net.fc, self.test_fgt_loader, opt.device)
            

            AUS = calculate_AUS(forgettest_val_acc, retaintest_val_acc, Aor)  # Calculate AUS using the accuracies

            # wandb.log({"retainfull_val_acc": retainfull_val_acc, "forgetfull_val_acc": forgetfull_val_acc,
            #            "retaintest_val_acc": retaintest_val_acc, "forgettest_val_acc": forgettest_val_acc, "AUS": AUS})

            aus_history.append(AUS)

            print(f"Epoch {epoch+1}/{opt.epochs_unlearn} | "
                f"Train Retain Acc: {retain_accuracy:.2f}% | Train Forget Acc: {forget_accuracy:.2f}% | "
                f"Val Retain full Acc: {retainfull_val_acc:.2f}% | Val Forget full Acc: {forgetfull_val_acc:.2f}%  | "
                f"Val Retain Test Acc: {retaintest_val_acc:.2f}% | Val Forget Test Acc: {forgettest_val_acc:.2f}% | "
                f"AUS: {AUS:.2f}"
                )

            # Update the best result
            if AUS > best_aus or (
                AUS == best_aus and (
                    retaintest_val_acc > best_retain_acc or
                    forgettest_val_acc < best_forget_acc
                )
            ):
                best_aus = max(best_aus, AUS)
                best_retain_acc = max(best_retain_acc, retaintest_val_acc)
                best_forget_acc = min(best_forget_acc, forgettest_val_acc)

                best_model_state = deepcopy(self.net.state_dict())


                checkpoint_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                )

                torch.save(best_model_state, checkpoint_path)
                print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={best_aus:.4f} to {checkpoint_path}")


                best_results = {
                    "Epoch": epoch + 1,
                    "Unlearning Train Retain Acc": round(retain_accuracy, 4),
                    "Unlearning Train Forget Acc": round(forget_accuracy, 4),
                    "Unlearning Val Retain Full Acc": round(retainfull_val_acc, 4),
                    "Unlearning Val Forget Full Acc": round(forgetfull_val_acc, 4),
                    "Unlearning Val Retain Test Acc": round(retaintest_val_acc, 4),
                    "Unlearning Val Forget Test Acc": round(forgettest_val_acc, 4),
                    "AUS": round(AUS, 4)
                }

            if forgettest_val_acc == 0.0:
                zero_acc_fgt_counter += 1
            else:
                zero_acc_fgt_counter = 0

            if zero_acc_fgt_counter >= zero_acc_patience:
                print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                break


            if len(aus_history) > opt.patience:
                recent_trend_aus = aus_history[-opt.patience:]

                # Condition 1: AUS is decreasing for 'patience' epochs
                decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(len(recent_trend_aus)-1))

                # Condition 2: AUS has not changed significantly for 'patience' epochs
                no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(len(recent_trend_aus)-1))


                if decreasing_aus or no_change_aus:
                    print(f"Early stopping triggered at epoch {epoch+1} due to AUS criteria.")
                    break

            # Additional early stopping: AUS < 0.4 for more than 20 epochs
            low_aus_threshold = 0.4
            low_aus_patience = 20

            low_aus_count = sum(a < low_aus_threshold for a in aus_history[-low_aus_patience:])
            if low_aus_count >= low_aus_patience:
                print(f"[Early Stopping] Triggered due to AUS < {low_aus_threshold} for {low_aus_patience} consecutive epochs.")
                break

            
            # === Save current epoch to CSV immediately ===
            log_epoch_to_csv(
                epoch=epoch,
                epoch_times=duration,
                train_retain_acc=round(retain_accuracy ,4),
                train_fgt_acc=round(forget_accuracy ,4),
                val_test_retain_acc=round(retaintest_val_acc ,4),
                val_test_fgt_acc=round(forgettest_val_acc ,4),
                val_full_retain_acc=round(retainfull_val_acc ,4),
                val_full_fgt_acc=round(forgetfull_val_acc ,4),
                AUS=round(AUS,4),
                mode=opt.method,
                dataset=opt.dataset,
                model=opt.model,
                class_to_remove=self.class_to_remove,
                seed=opt.seed,
                retain_count=retain_count,
                forget_count=forget_count,
                total_count=total_count)

        unlearning_time_until_best = sum(epoch_times[:best_results["Epoch"] + 1])

        log_summary_across_classes(
            best_epoch=round(best_results["Epoch"],4),
            train_retain_acc=round(best_results["Unlearning Train Retain Acc"],4),
            train_fgt_acc=round(best_results["Unlearning Train Forget Acc"],4),
            val_test_retain_acc=round(best_results["Unlearning Val Retain Test Acc"],4),
            val_test_fgt_acc=round(best_results["Unlearning Val Forget Test Acc"],4),
            val_full_retain_acc=round(best_results["Unlearning Val Retain Full Acc"],4),
            val_full_fgt_acc=round(best_results["Unlearning Val Forget Full Acc"],4),
            AUS=round(best_results["AUS"],4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))





        self.net.eval()
        return self.net


class newmethod(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        self.loader = self.train_fgt_loader
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)

    def loss_f(self, inputs, targets):
        with torch.no_grad():
            logits = self.head_fc(inputs)
            top2 = torch.topk(logits, k=2, dim=1)
            pred = top2.indices[:, 1]  # second-best prediction (nearest incorrect)
        outputs = self.head_fc(inputs)
        loss = self.criterion(outputs, pred)
        return loss


class BoundaryShrink(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader,
                 retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader,
                         test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        
        self.train_retain_loader = train_retain_loader
        self.train_fgt_loader = train_fgt_loader
        self.test_retain_loader = test_retain_loader
        self.test_fgt_loader = test_fgt_loader
        self.retainfull_loader_real = retainfull_loader_real
        self.forgetfull_loader_real = forgetfull_loader_real
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)

    def run(self):
       
        unlearn_model_fc = deepcopy(self.head_fc).to(opt.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(unlearn_model_fc.parameters(), lr=opt.lr_unlearn, momentum=0.9)

        set_classifier(self.net, unlearn_model_fc)

        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs
        
        aus_history = []  

        best_results = None
        best_aus = float('-inf')  # Maximum AUS
        best_retain_acc = float('-inf')  # Maximum retaintest_val_acc
        best_forget_acc = float('inf')  # Minimum forgettest_val_acc

        Aor = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count

        epoch_times = []

        for epoch in range(opt.epochs_unlearn):
            start_time = time.time()

            for batch_idx, (x, y) in enumerate(self.train_fgt_loader):
                x, y = x.to(opt.device), y.to(opt.device)

                # Nearest but incorrect class (2nd highest logit)
                with torch.no_grad():
                    logits = unlearn_model_fc(x)
                    top2 = torch.topk(logits, k=2, dim=1).indices
                    pred_label = torch.where(top2[:, 0] == y, top2[:, 1], top2[:, 0])

                unlearn_model_fc.train()
                optimizer.zero_grad()
                output = unlearn_model_fc(x)
                loss = criterion(output, pred_label)
                loss.backward()
                optimizer.step()

            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)
                
            retain_accuracy = evaluate_embedding_accuracy(unlearn_model_fc, self.train_retain_loader, opt.device)
            forget_accuracy = evaluate_embedding_accuracy(unlearn_model_fc, self.train_fgt_loader, opt.device)

            retainfull_val_acc = evaluate_embedding_accuracy(unlearn_model_fc, self.retainfull_loader_real, opt.device)
            forgetfull_val_acc = evaluate_embedding_accuracy(unlearn_model_fc, self.forgetfull_loader_real, opt.device)

            retaintest_val_acc = evaluate_embedding_accuracy(unlearn_model_fc, self.test_retain_loader, opt.device)
            forgettest_val_acc = evaluate_embedding_accuracy(unlearn_model_fc, self.test_fgt_loader, opt.device)
            

            AUS = calculate_AUS(forgettest_val_acc, retaintest_val_acc, Aor)  # Calculate AUS using the accuracies

            # wandb.log({"retainfull_val_acc": retainfull_val_acc, "forgetfull_val_acc": forgetfull_val_acc,
            #            "retaintest_val_acc": retaintest_val_acc, "forgettest_val_acc": forgettest_val_acc, "AUS": AUS})

            aus_history.append(AUS)

            print(f"Epoch {epoch+1}/{opt.epochs_unlearn} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Retain Acc: {retain_accuracy:.2f}% | Train Forget Acc: {forget_accuracy:.2f}% | "
                f"Val Retain full Acc: {retainfull_val_acc:.2f}% | Val Forget full Acc: {forgetfull_val_acc:.2f}%  | "
                f"Val Retain Test Acc: {retaintest_val_acc:.2f}% | Val Forget Test Acc: {forgettest_val_acc:.2f}% | "
                f"AUS: {AUS:.2f}"
                )

            # Update the best result
            if AUS > best_aus or (
                AUS == best_aus and (
                    retaintest_val_acc > best_retain_acc or
                    forgettest_val_acc < best_forget_acc
                )
            ):
                best_aus = max(best_aus, AUS)
                best_retain_acc = max(best_retain_acc, retaintest_val_acc)
                best_forget_acc = min(best_forget_acc, forgettest_val_acc)

                best_model_state = deepcopy(self.net.state_dict())

                checkpoint_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                )

                torch.save(best_model_state, checkpoint_path)
                print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={best_aus:.4f} to {checkpoint_path}")

                
                best_results = {
                    "Epoch": epoch + 1,
                    "Loss": round(loss.item(), 4),
                    "Unlearning Train Retain Acc": round(retain_accuracy, 4),
                    "Unlearning Train Forget Acc": round(forget_accuracy, 4),
                    "Unlearning Val Retain Full Acc": round(retainfull_val_acc, 4),
                    "Unlearning Val Forget Full Acc": round(forgetfull_val_acc, 4),
                    "Unlearning Val Retain Test Acc": round(retaintest_val_acc, 4),
                    "Unlearning Val Forget Test Acc": round(forgettest_val_acc, 4),
                    "AUS": round(AUS, 4)
                }

            if forgettest_val_acc == 0.0:
                zero_acc_fgt_counter += 1
            else:
                zero_acc_fgt_counter = 0

            if zero_acc_fgt_counter >= zero_acc_patience:
                print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                break


            if len(aus_history) > opt.patience:
                recent_trend_aus = aus_history[-opt.patience:]

                # Condition 1: AUS is decreasing for 'patience' epochs
                decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(len(recent_trend_aus)-1))

                # Condition 2: AUS has not changed significantly for 'patience' epochs
                no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(len(recent_trend_aus)-1))


                if decreasing_aus or no_change_aus:
                    print(f"Early stopping triggered at epoch {epoch+1} due to AUS criteria.")
                    break

            # Additional early stopping: AUS < 0.4 for more than 20 epochs
            low_aus_threshold = 0.4
            low_aus_patience = 20

            low_aus_count = sum(a < low_aus_threshold for a in aus_history[-low_aus_patience:])
            if low_aus_count >= low_aus_patience:
                print(f"[Early Stopping] Triggered due to AUS < {low_aus_threshold} for {low_aus_patience} consecutive epochs.")
                break

            
            # === Save current epoch to CSV immediately ===
            log_epoch_to_csv(
                epoch=epoch,
                epoch_times=duration,
                train_retain_acc=round(retain_accuracy,4),
                train_fgt_acc=round(forget_accuracy,4),
                val_test_retain_acc=round(retaintest_val_acc,4),
                val_test_fgt_acc=round(forgettest_val_acc,4),
                val_full_retain_acc=round(retainfull_val_acc,4),
                val_full_fgt_acc=round(forgetfull_val_acc,4),
                AUS=round(AUS,4),
                mode=opt.method,
                dataset=opt.dataset,
                model=opt.model,
                class_to_remove=self.class_to_remove,
                seed=opt.seed,
                retain_count=retain_count,
                forget_count=forget_count,
                total_count=total_count)

            
        unlearning_time_until_best = sum(epoch_times[:best_results["Epoch"] + 1])
                
        log_summary_across_classes(
            best_epoch=round(best_results["Epoch"],4),
            train_retain_acc=round(best_results["Unlearning Train Retain Acc"],4),
            train_fgt_acc=round(best_results["Unlearning Train Forget Acc"],4),
            val_test_retain_acc=round(best_results["Unlearning Val Retain Test Acc"],4),
            val_test_fgt_acc=round(best_results["Unlearning Val Forget Test Acc"],4),
            val_full_retain_acc=round(best_results["Unlearning Val Retain Full Acc"],4),
            val_full_fgt_acc=round(best_results["Unlearning Val Forget Full Acc"],4),
            AUS=round(best_results["AUS"],4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))


        self.model = self.net
        
        return self.model


class BoundaryExpanding(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader,
                 retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader,
                         test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)

        self.train_retain_loader = train_retain_loader
        self.train_fgt_loader = train_fgt_loader
        self.test_retain_loader = test_retain_loader
        self.test_fgt_loader = test_fgt_loader
        self.retainfull_loader_real = retainfull_loader_real
        self.forgetfull_loader_real = forgetfull_loader_real
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)

    def run(self):
      
        # Extract the last Linear layer from self.head_fc, whether it's Sequential or Linear
        def find_final_linear(module):
            if isinstance(module, nn.Linear):
                return module
            elif isinstance(module, nn.Sequential):
                for layer in reversed(module):
                    if isinstance(layer, nn.Linear):
                        return layer
            raise ValueError("No nn.Linear layer found in self.head_fc")

        final_linear = find_final_linear(self.head_fc)
        embedding_dim = final_linear.in_features
        num_classes = final_linear.out_features

        shadow_class = num_classes

        # Widen model with an extra output for the shadow class
        widen_model = nn.Linear(embedding_dim, num_classes + 1).to(opt.device)
        with torch.no_grad():
            widen_model.weight[:num_classes] = final_linear.weight
            widen_model.bias[:num_classes] = final_linear.bias

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(widen_model.parameters(), lr=opt.lr_unlearn, momentum=0.9)
            
        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs
        
        aus_history = []  

        best_results = None
        best_aus = float('-inf')  # Maximum AUS
        best_retain_acc = float('-inf')  # Maximum retaintest_val_acc
        best_forget_acc = float('inf')  # Minimum forgettest_val_acc

        Aor = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count

        epoch_times = []

        for epoch in range(opt.epochs_unlearn):
            start_time = time.time()

            for batch_idx, (x, y) in enumerate(self.train_fgt_loader):
                x, y = x.to(opt.device), y.to(opt.device)
                target = torch.full_like(y, fill_value=shadow_class)

                widen_model.train()
                optimizer.zero_grad()
                logits = widen_model(x)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)

            retain_accuracy = evaluate_embedding_accuracy(widen_model, self.train_retain_loader, opt.device)
            forget_accuracy = evaluate_embedding_accuracy(widen_model, self.train_fgt_loader, opt.device)

            retainfull_val_acc = evaluate_embedding_accuracy(widen_model, self.retainfull_loader_real, opt.device)
            forgetfull_val_acc = evaluate_embedding_accuracy(widen_model, self.forgetfull_loader_real, opt.device)

            retaintest_val_acc = evaluate_embedding_accuracy(widen_model, self.test_retain_loader, opt.device)
            forgettest_val_acc = evaluate_embedding_accuracy(widen_model, self.test_fgt_loader, opt.device)
            

            AUS = calculate_AUS(forgettest_val_acc, retaintest_val_acc, Aor)  # Calculate AUS using the accuracies

            # wandb.log({"retainfull_val_acc": retainfull_val_acc, "forgetfull_val_acc": forgetfull_val_acc,
            #            "retaintest_val_acc": retaintest_val_acc, "forgettest_val_acc": forgettest_val_acc, "AUS": AUS})

            aus_history.append(AUS)

            print(f"Epoch {epoch+1}/{opt.epochs_unlearn} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Retain Acc: {retain_accuracy:.2f}% | Train Forget Acc: {forget_accuracy:.2f}% | "
                f"Val Retain full Acc: {retainfull_val_acc:.2f}% | Val Forget full Acc: {forgetfull_val_acc:.2f}%  | "
                f"Val Retain Test Acc: {retaintest_val_acc:.2f}% | Val Forget Test Acc: {forgettest_val_acc:.2f}% | "
                f"AUS: {AUS:.2f}"
                )

            # Update the best result
            if AUS > best_aus or (
                AUS == best_aus and (
                    retaintest_val_acc > best_retain_acc or
                    forgettest_val_acc < best_forget_acc
                )
            ):
                best_aus = max(best_aus, AUS)
                best_retain_acc = max(best_retain_acc, retaintest_val_acc)
                best_forget_acc = min(best_forget_acc, forgettest_val_acc)

                with torch.no_grad():
                    head = get_classifier(self.net)
                    def find_final_linear(module):
                        if isinstance(module, nn.Linear):
                            return module
                        if isinstance(module, nn.Sequential):
                            for layer in reversed(module):
                                if isinstance(layer, nn.Linear):
                                    return layer
                        # generic fallback: walk all submodules, pick the last nn.Linear
                        last_lin = None
                        for m in module.modules():
                            if isinstance(m, nn.Linear):
                                last_lin = m
                        if last_lin is None:
                            raise ValueError("No nn.Linear layer found in classifier head")
                        return last_lin

                    last_linear = find_final_linear(head)
                    last_linear.weight.copy_(widen_model.weight[:num_classes])
                    if last_linear.bias is not None and widen_model.bias is not None:
                        last_linear.bias.copy_(widen_model.bias[:num_classes])

                    best_model_state = deepcopy(self.net.state_dict())


                    checkpoint_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                    )

                    torch.save(best_model_state, checkpoint_path)
                    print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={best_aus:.4f} to {checkpoint_path}")

                    best_results = {
                        "Epoch": epoch + 1,
                        "Loss": round(loss.item(), 4),
                        "Unlearning Train Retain Acc": round(retain_accuracy, 4),
                        "Unlearning Train Forget Acc": round(forget_accuracy, 4),
                        "Unlearning Val Retain Full Acc": round(retainfull_val_acc, 4),
                        "Unlearning Val Forget Full Acc": round(forgetfull_val_acc, 4),
                        "Unlearning Val Retain Test Acc": round(retaintest_val_acc, 4),
                        "Unlearning Val Forget Test Acc": round(forgettest_val_acc, 4),
                        "AUS": round(AUS, 4)
                    }

            if forgettest_val_acc == 0.0:
                zero_acc_fgt_counter += 1
            else:
                zero_acc_fgt_counter = 0

            if zero_acc_fgt_counter >= zero_acc_patience:
                print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                break


            if len(aus_history) > opt.patience:
                recent_trend_aus = aus_history[-opt.patience:]

                # Condition 1: AUS is decreasing for 'patience' epochs
                decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(len(recent_trend_aus)-1))

                # Condition 2: AUS has not changed significantly for 'patience' epochs
                no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(len(recent_trend_aus)-1))


                if decreasing_aus or no_change_aus:
                    print(f"Early stopping triggered at epoch {epoch+1} due to AUS criteria.")
                    break

            # Additional early stopping: AUS < 0.4 for more than 20 epochs
            low_aus_threshold = 0.4
            low_aus_patience = 20

            low_aus_count = sum(a < low_aus_threshold for a in aus_history[-low_aus_patience:])
            if low_aus_count >= low_aus_patience:
                print(f"[Early Stopping] Triggered due to AUS < {low_aus_threshold} for {low_aus_patience} consecutive epochs.")
                break

            # === Save current epoch to CSV immediately ===
            log_epoch_to_csv(
                epoch=epoch,
                epoch_times=duration,
                train_retain_acc=round(retain_accuracy / 100,4),
                train_fgt_acc=round(forget_accuracy / 100,4),
                val_test_retain_acc=round(retaintest_val_acc / 100,4),
                val_test_fgt_acc=round(forgettest_val_acc / 100,4),
                val_full_retain_acc=round(retainfull_val_acc / 100,4),
                val_full_fgt_acc=round(forgetfull_val_acc / 100,4),
                AUS=round(AUS,4),
                mode=opt.method,
                dataset=opt.dataset,
                model=opt.model,
                class_to_remove=self.class_to_remove,
                seed=opt.seed,
                retain_count=retain_count,
                forget_count=forget_count,
                total_count=total_count)
            
        unlearning_time_until_best = sum(epoch_times[:best_results["Epoch"] + 1])

        log_summary_across_classes(
            best_epoch=round(best_results["Epoch"],4),
            train_retain_acc=round(best_results["Unlearning Train Retain Acc"] / 100,4),
            train_fgt_acc=round(best_results["Unlearning Train Forget Acc"] / 100,4),
            val_test_retain_acc=round(best_results["Unlearning Val Retain Test Acc"] / 100,4),
            val_test_fgt_acc=round(best_results["Unlearning Val Forget Test Acc"] / 100,4),
            val_full_retain_acc=round(best_results["Unlearning Val Retain Full Acc"] / 100,4),
            val_full_fgt_acc=round(best_results["Unlearning Val Forget Full Acc"] / 100,4),
            AUS=round(best_results["AUS"],4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))

                        
        # Prune the shadow class to return a normal classifier
        # Final pruning before return
        with torch.no_grad():
            self.net.fc.weight.copy_(widen_model.weight[:num_classes])
            self.net.fc.bias.copy_(widen_model.bias[:num_classes])

        self.model = self.net
        
        return self.model

class SCRUB(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)        

        self.teacher = deepcopy(net).to(opt.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.student = deepcopy(net).to(opt.device)  # Clone of the original FC layer
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)

    def run(self):
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

  
        # Compute Aor from original model (accuracy on retain test set)
        Aor = calculate_accuracy(self.teacher, self.test_retain_loader, use_fc_only=True)

        # Load all features/labels from loaders
        def flatten_loader(loader):
            features, labels = [], []
            for x, y in loader:
                features.append(x)
                labels.append(y)
            return torch.cat(features), torch.cat(labels)

        retain_features_train, retain_labels_train = flatten_loader(self.train_retain_loader)
        forget_features_train, forget_labels_train = flatten_loader(self.train_fgt_loader)

        retainfull_loader_val=self.retainfull_loader_real
        forgetfull_loader_val=self.forgetfull_loader_real
        retaintest_loader_val=self.test_retain_loader
        forgettest_loader_val=self.test_fgt_loader
        alpha=0.45
        gamma=0.45
        betha=0.1
    
        retain_synth_loader_train = DataLoader(TensorDataset(retain_features_train, retain_labels_train), batch_size=opt.batch_size, shuffle=True)
        forget_synth_loader_train = DataLoader(TensorDataset(forget_features_train, forget_labels_train), batch_size=opt.batch_size, shuffle=True)

        teacher_fc = get_classifier(self.teacher).to(opt.device)
        student_fc = get_classifier(self.student).to(opt.device)
        optimizer = optim.Adam(student_fc.parameters(), lr=opt.lr_unlearn)
        loss_ce = nn.CrossEntropyLoss()
        
        for param in teacher_fc.parameters():
            param.requires_grad = False
        
        for param in student_fc.parameters():
            param.requires_grad = True
            
        for name, param in teacher_fc.named_parameters():
            print(f"Teacher {name}: requires_grad={param.requires_grad}")

        for name, param in student_fc.named_parameters():
            print(f"student {name}: requires_grad={param.requires_grad}")

        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs
        
        results = []
        aus_history = []  

        best_results = None
        best_aus = float('-inf')  # Maximum AUS
        best_retain_acc = float('-inf')  # Maximum retaintest_val_acc
        best_forget_acc = float('inf')  # Minimum forgettest_val_acc

        # Evaluate student model before training (Epoch 0)
        student_fc.eval()

        retain_accuracy = evaluate_embedding_accuracy(student_fc, retain_synth_loader_train, opt.device)
        forget_accuracy = evaluate_embedding_accuracy(student_fc, forget_synth_loader_train, opt.device)

        retainfull_val_acc = evaluate_embedding_accuracy(student_fc, retainfull_loader_val, opt.device)
        forgetfull_val_acc = evaluate_embedding_accuracy(student_fc, forgetfull_loader_val, opt.device)

        retaintest_val_acc = evaluate_embedding_accuracy(student_fc, retaintest_loader_val, opt.device)
        forgettest_val_acc = evaluate_embedding_accuracy(student_fc, forgettest_loader_val, opt.device)

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

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        
        epoch_times = []

        for epoch in range(opt.epochs_unlearn):
            start_time = time.time()

            student_fc.train()
            optimizer.zero_grad()

            retain_logits_student = student_fc(retain_features_train)
            forget_logits_student = student_fc(forget_features_train) 
            
            with torch.no_grad():
                retain_logits_teacher = teacher_fc(retain_features_train) 
                forget_logits_teacher = teacher_fc(forget_features_train)
            
            # Compute Losses
            loss_kd_retain = kd_loss(retain_logits_student, retain_logits_teacher)

            loss_kd_forget = -kd_loss(forget_logits_student, forget_logits_teacher)

            loss_ce_retain = loss_ce(retain_logits_student, retain_labels_train)
            loss_ce_forget = loss_ce(forget_logits_student, forget_labels_train)

            # Total loss
            loss = (alpha * loss_kd_retain) + (gamma * loss_ce_retain) + (betha * loss_kd_forget)

            # Backpropagation
            loss.backward()
            optimizer.step()
                        
            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)
            
            student_fc.eval()
            
            retain_accuracy = evaluate_embedding_accuracy(student_fc, retain_synth_loader_train, opt.device)
            forget_accuracy = evaluate_embedding_accuracy(student_fc, forget_synth_loader_train, opt.device)

            retainfull_val_acc = evaluate_embedding_accuracy(student_fc, retainfull_loader_val, opt.device)
            forgetfull_val_acc = evaluate_embedding_accuracy(student_fc, forgetfull_loader_val, opt.device)

            retaintest_val_acc = evaluate_embedding_accuracy(student_fc, retaintest_loader_val, opt.device)
            forgettest_val_acc = evaluate_embedding_accuracy(student_fc, forgettest_loader_val, opt.device)
            
            AUS = calculate_AUS(forgettest_val_acc, retaintest_val_acc, Aor)  # Calculate AUS using the accuracies

            # wandb.log({"retainfull_val_acc": retainfull_val_acc, "forgetfull_val_acc": forgetfull_val_acc,
            #            "retaintest_val_acc": retaintest_val_acc, "forgettest_val_acc": forgettest_val_acc, "AUS": AUS})

            aus_history.append(AUS)

            print(f"Epoch {epoch+1}/{opt.epochs_unlearn} | "
                f"Loss: {loss.item():.4f} | "
                f"Retain CE Loss: {loss_ce_retain.item():.4f} | Forget CE Loss: {loss_ce_forget.item():.4f} | "
                f"Retain KD Loss: {loss_kd_retain.item():.4f} | Forget KD Loss: {loss_kd_forget.item():.4f} | "
                f"Train Retain Acc: {retain_accuracy:.2f}% | Train Forget Acc: {forget_accuracy:.2f}% | "
                f"Val Retain full Acc: {retainfull_val_acc:.2f}% | Val Forget full Acc: {forgetfull_val_acc:.2f}%  | "
                f"Val Retain Test Acc: {retaintest_val_acc:.2f}% | Val Forget Test Acc: {forgettest_val_acc:.2f}% | "
                f"AUS: {AUS:.2f}"
                )

            # Update the best result
            if AUS > best_aus or (
                AUS == best_aus and (
                    retaintest_val_acc > best_retain_acc or
                    forgettest_val_acc < best_forget_acc
                )
            ):
                best_aus = max(best_aus, AUS)
                best_retain_acc = max(best_retain_acc, retaintest_val_acc)
                best_forget_acc = min(best_forget_acc, forgettest_val_acc)

                self.student.fc = student_fc

                best_model_state = deepcopy(self.student.state_dict())

                checkpoint_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                )

                torch.save(best_model_state, checkpoint_path)
                print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={best_aus:.4f} to {checkpoint_path}")

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

            if forgettest_val_acc == 0.0:
                zero_acc_fgt_counter += 1
            else:
                zero_acc_fgt_counter = 0

            if zero_acc_fgt_counter >= zero_acc_patience:
                print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                break


            if len(aus_history) > opt.patience:
                recent_trend_aus = aus_history[-opt.patience:]

                # Condition 1: AUS is decreasing for 'patience' epochs
                decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(len(recent_trend_aus)-1))

                # Condition 2: AUS has not changed significantly for 'patience' epochs
                no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(len(recent_trend_aus)-1))


                if decreasing_aus or no_change_aus:
                    print(f"Early stopping triggered at epoch {epoch+1} due to AUS criteria.")
                    break

            # Additional early stopping: AUS < 0.4 for more than 20 epochs
            low_aus_threshold = 0.4
            low_aus_patience = 20

            low_aus_count = sum(a < low_aus_threshold for a in aus_history[-low_aus_patience:])
            if low_aus_count >= low_aus_patience:
                print(f"[Early Stopping] Triggered due to AUS < {low_aus_threshold} for {low_aus_patience} consecutive epochs.")
                break

            # === Save current epoch to CSV immediately ===
            log_epoch_to_csv(
                epoch=epoch,
                epoch_times=duration,
                train_retain_acc=round(retain_accuracy,4),
                train_fgt_acc=round(forget_accuracy,4),
                val_test_retain_acc=round(retaintest_val_acc,4),
                val_test_fgt_acc=round(forgettest_val_acc,4),
                val_full_retain_acc=round(retainfull_val_acc,4),
                val_full_fgt_acc=round(forgetfull_val_acc,4),
                AUS=round(AUS,4),
                mode=opt.method,
                dataset=opt.dataset,
                model=opt.model,
                class_to_remove=self.class_to_remove,
                seed=opt.seed,
                retain_count=retain_count,
                forget_count=forget_count,
                total_count=total_count)
            
                
        unlearning_time_until_best = sum(epoch_times[:best_results["Epoch"] + 1])

        log_summary_across_classes(
            best_epoch=round(best_results["Epoch"],4),
            train_retain_acc=round(best_results["Unlearning Train Retain Acc"] ,4),
            train_fgt_acc=round(best_results["Unlearning Train Forget Acc"],4),
            val_test_retain_acc=round(best_results["Unlearning Val Retain Test Acc"],4),
            val_test_fgt_acc=round(best_results["Unlearning Val Forget Test Acc"],4),
            val_full_retain_acc=round(best_results["Unlearning Val Retain Full Acc"],4),
            val_full_fgt_acc=round(best_results["Unlearning Val Forget Full Acc"],4),
            AUS=round(best_results["AUS"],4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))

        
        self.student.fc = student_fc

        return self.student


class DUCK(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)        
        self.class_to_remove = class_to_remove
        self.head_fc = get_classifier(self.net)
    
        
    def pairwise_cos_dist(self, x, y):
        """Compute pairwise cosine distance between two tensors"""
        x_norm = torch.norm(x, dim=1).unsqueeze(1)
        y_norm = torch.norm(y, dim=1).unsqueeze(1)
        x = x / x_norm
        y = y / y_norm
        return 1 - torch.mm(x, y.transpose(0, 1))


    def run(self):
        """compute embeddings"""
        #lambda1 fgt
        #lambda2 retain

        # Freeze all model params except final layer
        for param in self.net.parameters():
            param.requires_grad = False

        for param in self.head_fc.parameters():
            param.requires_grad = True
            
        if opt.model == 'AllCNN':
            self.fc = self.net.classifier
        else:
            self.fc = self.net.fc

        self.net.eval()  # just in case
        self.fc.train()

        # === Get class centroids from retain embeddings
        retain_embs, retain_labels = [], []
        for emb, label in self.train_retain_loader:
            retain_embs.append(emb.to(opt.device))
            retain_labels.append(label.to(opt.device))
        retain_embs = torch.cat(retain_embs)
        retain_labels = torch.cat(retain_labels)
        
        
        # compute distribs from embeddings
        distribs=[]
        for i in range(opt.num_classes):
            if type(self.class_to_remove) is list:
                if i not in self.class_to_remove:
                    distribs.append(retain_embs[retain_labels==i].mean(0))
            else:
                distribs.append(retain_embs[retain_labels==i].mean(0))
        distribs=torch.stack(distribs)


        optimizer = optim.Adam(self.head_fc.parameters(), lr=opt.lr_unlearn, weight_decay=opt.wd_unlearn)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.scheduler, gamma=0.5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if opt.dataset == 'TinyImageNet' else 0)

        best_aus = -float('inf')
        best_epoch = -1
        patience_counter = 0
        patience = opt.patience
        
        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs
        
        aus_history = []
        a_or_value = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)

        init = True
        all_closest_distribs = []
        if opt.dataset == 'TinyImageNet':
            ls = 0.2
        else:
            ls = 0
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        
        epoch_times = []

        print('Num batch forget: ',len(self.train_fgt_loader), 'Num batch retain: ',len(self.train_retain_loader))
        for epoch in tqdm(range(opt.epochs_unlearn)):
            start_time = time.time()

            for n_batch, (emb_fgt, lab_fgt) in enumerate(self.train_fgt_loader):
                for n_batch_ret, (emb_ret, lab_ret) in enumerate(self.train_retain_loader):
                    emb_ret, lab_ret,emb_fgt, lab_fgt  = emb_ret.to(opt.device), lab_ret.to(opt.device),emb_fgt.to(opt.device), lab_fgt.to(opt.device)
                    
                    optimizer.zero_grad()

                    # compute pairwise cosine distance between embeddings and distribs
                    dists = self.pairwise_cos_dist(emb_fgt, distribs)

                    if init:
                        closest_distribs = torch.argsort(dists, dim=1)
                        tmp = closest_distribs[:, 0]
                        closest_distribs = torch.where(tmp == lab_fgt, closest_distribs[:, 1], tmp)
                        all_closest_distribs.append(closest_distribs)
                        closest_distribs = all_closest_distribs[-1]
                    else:
                        closest_distribs = all_closest_distribs[n_batch]

                    dists = dists[torch.arange(dists.shape[0]), closest_distribs[:dists.shape[0]]]
                    loss_fgt = torch.mean(dists) * opt.lambda_1

                    outputs_ret = self.fc(emb_ret)

                    loss_ret = criterion(outputs_ret/opt.temperature, lab_ret)*opt.lambda_2
                    loss = loss_ret+ loss_fgt
                    
                    if n_batch_ret>opt.num_retain_samp:
                        del loss,loss_ret,loss_fgt, emb_fgt, emb_ret, outputs_ret,dists
                        break
                    
                    loss.backward()
                    optimizer.step()

            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)



            # evaluate accuracy on forget set every batch
            with torch.no_grad():
                self.net.eval()
                acc_train_ret = calculate_accuracy(self.net, self.train_retain_loader, use_fc_only=True)
                acc_train_fgt = calculate_accuracy(self.net, self.train_fgt_loader, use_fc_only=True)
                acc_test_val_ret = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
                acc_test_val_fgt = calculate_accuracy(self.net, self.test_fgt_loader, use_fc_only=True)
                acc_full_val_ret = calculate_accuracy(self.net, self.retainfull_loader_real, use_fc_only=True)
                acc_full_val_fgt = calculate_accuracy(self.net, self.forgetfull_loader_real, use_fc_only=True)


                a_t = Complex(acc_test_val_ret, 0.0)
                a_f = Complex(acc_test_val_fgt, 0.0)
                a_or = Complex(a_or_value, 0.0)

                aus_result = AUS(a_t, a_or, a_f)
                aus_value = aus_result.value
                aus_error = aus_result.error

                aus_history.append(aus_value)
                
                self.net.train()
                print(f"Train Retain Acc: {acc_train_ret:.3f},"
                      f"Train Forget Acc: {acc_train_fgt:.3f},"
                      f"Val Retain Test Acc: {acc_test_val_ret:.3f},"
                      f"Val Forget Test Acc: {acc_test_val_fgt:.3f},"
                      f"Val Retain Full Acc: {acc_full_val_ret:.3f},"
                      f"Val Forget Full Acc: {acc_full_val_fgt:.3f},"
                      f"target Acc: {self.target_accuracy:.3f},"
                      f"AUS: {aus_value:.3f}±{aus_error:.4f}")

                    
                if aus_value > best_aus:
                    best_aus = aus_value
                    best_model_state = deepcopy(self.net.state_dict())
                    best_epoch = epoch

                    best_acc_train_ret = acc_train_ret
                    best_acc_train_fgt = acc_train_fgt
                    best_acc_test_val_ret = acc_test_val_ret
                    best_acc_test_val_fgt = acc_test_val_fgt
                    best_acc_full_val_ret = acc_full_val_ret
                    best_acc_full_val_fgt = acc_full_val_fgt
                    

                if acc_test_val_fgt == 0.0:
                    zero_acc_fgt_counter += 1
                else:
                    zero_acc_fgt_counter = 0

                if zero_acc_fgt_counter >= zero_acc_patience:
                    print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                    break
    
                if len(aus_history) > patience:
                    recent_trend_aus = aus_history[-patience:]

                    # Condition 1: AUS is decreasing
                    decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(patience - 1))

                    # Condition 2: AUS has not changed significantly
                    no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(patience - 1))

                    # Condition 3: AUS is above the minimum threshold
                    #aus_high_enough = aus_value >= 70

                    if (decreasing_aus or no_change_aus):# and aus_high_enough:
                        print(f"[Early Stopping] Triggered at epoch {epoch+1} due to AUS trend.")
                        break

                # Additional early stopping: AUS < 0.4 for more than 20 epochs
                low_aus_threshold = 0.4
                low_aus_patience = 20

                low_aus_count = sum(a < low_aus_threshold for a in aus_history[-low_aus_patience:])
                if low_aus_count >= low_aus_patience:
                    print(f"[Early Stopping] Triggered due to AUS < {low_aus_threshold} for {low_aus_patience} consecutive epochs.")
                    break

                log_epoch_to_csv(
                    epoch=epoch,
                    epoch_times=duration,
                    train_retain_acc=round(acc_train_ret, 4),
                    train_fgt_acc=round(acc_train_fgt, 4),
                    val_test_retain_acc=round(acc_test_val_ret, 4),
                    val_test_fgt_acc=round(acc_test_val_fgt, 4),
                    val_full_retain_acc=round(acc_full_val_ret, 4),
                    val_full_fgt_acc=round(acc_full_val_fgt, 4),
                    AUS=round(aus_value, 4),
                    mode=opt.method,
                    dataset=opt.dataset,
                    model=opt.model,
                    class_to_remove=self.class_to_remove,
                    seed=opt.seed,
                    retain_count=retain_count,
                    forget_count=forget_count,
                    total_count=total_count)

            init = False
            scheduler.step()
            
            
        unlearning_time_until_best = sum(epoch_times[:best_epoch + 1])

        log_summary_across_classes(
            best_epoch=best_epoch,
            train_retain_acc=round(best_acc_train_ret, 4),
            train_fgt_acc=round(best_acc_train_fgt, 4),
            val_test_retain_acc=round(best_acc_test_val_ret, 4),
            val_test_fgt_acc=round(best_acc_test_val_fgt, 4),
            val_full_retain_acc=round(best_acc_full_val_ret, 4),
            val_full_fgt_acc=round(best_acc_full_val_fgt, 4),
            AUS=round(best_aus, 4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))
        

        self.net.eval()
        return self.net


class RetrainedEmbedding(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)

        head = get_classifier(net)
        final_linear = None
        for m in (head.modules() if hasattr(head, "modules") else [head]):
            if isinstance(m, nn.Linear):
                final_linear = m
        if final_linear is None:
            raise RuntimeError("No nn.Linear found in classifier head")
        in_features = final_linear.in_features

        self.fc_layer = nn.Linear(in_features, opt.num_classes).to(opt.device)

        # if isinstance(net.fc, nn.Sequential):
        #     in_features = net.fc[-1].in_features
        # else:
        #     in_features = net.fc.in_features

        #self.fc_layer = nn.Linear(in_features, opt.num_classes).to(opt.device)
     
        self.optimizer = optim.SGD(self.fc_layer.parameters(), lr=opt.lr_unlearn, weight_decay=5e-5)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.4)

        if opt.dataset == 'TinyImageNet':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.1) #learning rate decay
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)


        self.class_to_remove = class_to_remove
        self.epochs = opt.epochs_unlearn


    def run(self):
                            
        """ Train the FC layer from embeddings """
        best_acc = 0.0
        patience_counter = 0
        best_epoch = -1
        best_train_acc = 0.0
        best_train_loss = 0.0
        best_val_loss = 0.0

        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs
        
        aus_history = []  

        best_results = None
        best_aus = float('-inf')  # Maximum AUS
        best_retain_acc = float('-inf')  # Maximum retaintest_val_acc
        best_forget_acc = float('inf')  # Minimum forgettest_val_acc

        Aor = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        epoch_times = []

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()

            self.fc_layer.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for embeddings, labels in self.train_retain_loader:
                embeddings, labels = embeddings.cuda(), labels.cuda()
                self.optimizer.zero_grad()
                outputs = self.fc_layer(embeddings)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)
    
            #train_loss = running_loss / len(self.train_retain_loader)
            #train_acc = 100. * correct / total
            self.scheduler.step()

            #val_acc, val_loss = self.evaluate(self.test_retain_loader)

            ## Save best model
            #if val_acc > best_acc:
            #    best_acc = val_acc
            #    best_epoch = epoch
            #    best_train_acc = train_acc
            #    best_train_loss = train_loss
            #    best_val_loss = val_loss
            #    patience_counter = 0
            #    #self.save_model()
            #    #print(f"New best model saved at epoch {epoch+1} with Val Acc: {best_acc:.2f}%")
            #else:
            #    patience_counter += 1
#
            #print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Train Loss {train_loss:.4f}, Val Acc {val_acc:.2f}%, Val Loss {val_loss:.4f}")
#
            #if patience_counter >= self.patience:
            #    print("Early stopping triggered.")
            #    break
#
            retain_accuracy = evaluate_embedding_accuracy(self.fc_layer, self.train_retain_loader, opt.device)
            forget_accuracy = evaluate_embedding_accuracy(self.fc_layer, self.train_fgt_loader, opt.device)

            retainfull_val_acc = evaluate_embedding_accuracy(self.fc_layer, self.retainfull_loader_real, opt.device)
            forgetfull_val_acc = evaluate_embedding_accuracy(self.fc_layer, self.forgetfull_loader_real, opt.device)

            retaintest_val_acc = evaluate_embedding_accuracy(self.fc_layer, self.test_retain_loader, opt.device)
            forgettest_val_acc = evaluate_embedding_accuracy(self.fc_layer, self.test_fgt_loader, opt.device)
            
            AUS = calculate_AUS(forgettest_val_acc, retaintest_val_acc, Aor)  # Calculate AUS using the accuracies

            # wandb.log({"retainfull_val_acc": retainfull_val_acc, "forgetfull_val_acc": forgetfull_val_acc,
            #            "retaintest_val_acc": retaintest_val_acc, "forgettest_val_acc": forgettest_val_acc, "AUS": AUS})

            aus_history.append(AUS)

            print(f"Epoch {epoch+1}/{opt.epochs_unlearn} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Retain Acc: {retain_accuracy:.2f}% | Train Forget Acc: {forget_accuracy:.2f}% | "
                f"Val Retain full Acc: {retainfull_val_acc:.2f}% | Val Forget full Acc: {forgetfull_val_acc:.2f}%  | "
                f"Val Retain Test Acc: {retaintest_val_acc:.2f}% | Val Forget Test Acc: {forgettest_val_acc:.2f}% | "
                f"AUS: {AUS:.2f}"
                )

            # Update the best result
            if AUS > best_aus or (
                AUS == best_aus and (
                    retaintest_val_acc > best_retain_acc or
                    forgettest_val_acc < best_forget_acc
                )
            ):
                best_aus = max(best_aus, AUS)
                best_retain_acc = max(best_retain_acc, retaintest_val_acc)
                best_forget_acc = min(best_forget_acc, forgettest_val_acc)
                
                self.net.fc = self.fc_layer
                best_model_state = deepcopy(self.net.state_dict())

                checkpoint_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                )

                torch.save(best_model_state, checkpoint_path)
                print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={best_aus:.4f} to {checkpoint_path}")
                                        
                best_results = {
                    "Epoch": epoch + 1,
                    "Loss": round(loss.item(), 4),
                    "Unlearning Train Retain Acc": round(retain_accuracy, 4),
                    "Unlearning Train Forget Acc": round(forget_accuracy, 4),
                    "Unlearning Val Retain Full Acc": round(retainfull_val_acc, 4),
                    "Unlearning Val Forget Full Acc": round(forgetfull_val_acc, 4),
                    "Unlearning Val Retain Test Acc": round(retaintest_val_acc, 4),
                    "Unlearning Val Forget Test Acc": round(forgettest_val_acc, 4),
                    "AUS": round(AUS, 4)
                }

            if forgettest_val_acc == 0.0:
                zero_acc_fgt_counter += 1
            else:
                zero_acc_fgt_counter = 0

            if zero_acc_fgt_counter >= zero_acc_patience:
                print(f"[Early Stopping] acc_test_fgt was 0 for {zero_acc_patience} consecutive epochs. Stopping...")
                break

            if len(aus_history) > opt.patience:
                recent_trend_aus = aus_history[-opt.patience:]

                # Condition 1: AUS is decreasing for 'patience' epochs
                decreasing_aus = all(recent_trend_aus[i] > recent_trend_aus[i+1] for i in range(len(recent_trend_aus)-1))

                # Condition 2: AUS has not changed significantly for 'patience' epochs
                no_change_aus = all(abs(recent_trend_aus[i] - recent_trend_aus[i+1]) < 1e-4 for i in range(len(recent_trend_aus)-1))


                if decreasing_aus or no_change_aus:
                    print(f"Early stopping triggered at epoch {epoch+1} due to AUS criteria.")
                    break
            
            # === Save current epoch to CSV immediately ===
            log_epoch_to_csv(
                epoch=epoch,
                epoch_times=duration,
                train_retain_acc=round(retain_accuracy,4),
                train_fgt_acc=round(forget_accuracy,4),
                val_test_retain_acc=round(retaintest_val_acc,4),
                val_test_fgt_acc=round(forgettest_val_acc,4),
                val_full_retain_acc=round(retainfull_val_acc,4),
                val_full_fgt_acc=round(forgetfull_val_acc,4),
                AUS=round(AUS,4),
                mode=opt.method,
                dataset=opt.dataset,
                model=opt.model,
                class_to_remove=self.class_to_remove,
                seed=opt.seed,
                retain_count=retain_count,
                forget_count=forget_count,
                total_count=total_count)
                            
        unlearning_time_until_best = sum(epoch_times[:best_results["Epoch"] + 1])

        log_summary_across_classes(
            best_epoch=round(best_results["Epoch"],4),
            train_retain_acc=round(best_results["Unlearning Train Retain Acc"],4),
            train_fgt_acc=round(best_results["Unlearning Train Forget Acc"],4),
            val_test_retain_acc=round(best_results["Unlearning Val Retain Test Acc"],4),
            val_test_fgt_acc=round(best_results["Unlearning Val Forget Test Acc"],4),
            val_full_retain_acc=round(best_results["Unlearning Val Retain Full Acc"],4),
            val_full_fgt_acc=round(best_results["Unlearning Val Forget Full Acc"],4),
            AUS=round(best_results["AUS"],4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))

        print(f"Best Epoch {best_epoch+1}: Train Acc {best_train_acc:.2f}%, Val Acc {best_acc:.2f}%")

        set_classifier(self.net, self.fc_layer)
        self.net.eval()

        return self.net
    
    
class LAU(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        
        self.teacher = deepcopy(self.net.fc).to(opt.device)
        self.student = self.net.fc  # Only modify fc layer

        self.class_to_remove = class_to_remove

     
        self.criterion_ce = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.student.parameters(), lr=opt.lr_unlearn, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.scheduler, gamma=0.5)

        self.alpha = 0.5
        self.temperature = 2.0
        self.epsilon = 4/255
        self.step_size = 1/255
        self.attack_steps = 7
        self.head_fc = get_classifier(self.net)


    def partial_pgd_attack(self, features, labels):
        adv_features = features.detach().clone()
        adv_features.requires_grad = True

        for _ in range(self.attack_steps):
            logits = self.student(adv_features)
            loss = self.criterion_ce(logits, labels)
            grad = torch.autograd.grad(loss, adv_features, retain_graph=False, create_graph=False)[0]

            adv_features = adv_features + self.step_size * grad.sign()
            delta = torch.clamp(adv_features - features, min=-self.epsilon, max=self.epsilon)
            adv_features = torch.clamp(features + delta, min=-1, max=1).detach()
            adv_features.requires_grad = True

        return adv_features.detach()

    def double_softmax_kd_loss(self, logits_student, logits_teacher, labels):
        # Conditional KD logic from Eq (4-6) in the paper
        pred_student = logits_student.argmax(dim=1)
        match = pred_student.eq(labels)  # (B,) boolean

        # Compute double softmax target
        soft_teacher = F.softmax(logits_teacher / self.temperature, dim=1)  # shape: (B, C)
        soft_student = F.softmax(logits_student / self.temperature, dim=1)

        mixed_target = torch.where(
            match.unsqueeze(1),
            soft_teacher,  # Use teacher if student prediction is correct
            soft_student   # Otherwise use student output as fallback
        )

        log_student = F.log_softmax(logits_student / self.temperature, dim=1)
        loss_kd = F.kl_div(log_student, mixed_target, reduction='batchmean') * (self.temperature ** 2)
        return loss_kd

    def run(self):
        best_aus = -float('inf')
        best_epoch = -1
        aus_history = []
        a_or_value = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
        zero_acc_fgt_counter = 0
        zero_acc_patience = 1000

        self.teacher.eval()

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        epoch_times = []

        for epoch in tqdm(range(opt.epochs_unlearn)):
            start_time = time.time()

            for inputs, targets in self.train_fgt_loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                
                adv_inputs = self.partial_pgd_attack(inputs, targets)
                
                logits_student = self.student(adv_inputs)
                with torch.no_grad():
                    logits_teacher = self.teacher(inputs)

                loss_ce = self.criterion_ce(logits_student, targets)
                loss_kd = self.double_softmax_kd_loss(logits_student, logits_teacher, targets)
                loss = (1 - self.alpha) * loss_ce + self.alpha * loss_kd  # T^2 is inside kd_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)

            with torch.no_grad():
                self.net.eval()
                acc_train_ret = calculate_accuracy(self.net, self.train_retain_loader, use_fc_only=True)
                acc_train_fgt = calculate_accuracy(self.net, self.train_fgt_loader, use_fc_only=True)
                acc_test_val_ret = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
                acc_test_val_fgt = calculate_accuracy(self.net, self.test_fgt_loader, use_fc_only=True)
                acc_full_val_ret = calculate_accuracy(self.net, self.retainfull_loader_real, use_fc_only=True)
                acc_full_val_fgt = calculate_accuracy(self.net, self.forgetfull_loader_real, use_fc_only=True)

                a_t = Complex(acc_test_val_ret, 0.0)
                a_f = Complex(acc_test_val_fgt, 0.0)
                a_or = Complex(a_or_value, 0.0)
                aus_result = AUS(a_t, a_or, a_f)
                aus_value = aus_result.value
                aus_history.append(aus_value)

                print(f"[Epoch {epoch}] Train Retain Acc: {acc_train_ret:.2f}%, Train Forget Acc: {acc_train_fgt:.2f}%, Val Retain Acc: {acc_test_val_ret:.2f}%, Val Forget Acc: {acc_test_val_fgt:.2f}%, AUS: {aus_value:.2f}")

                if aus_value > best_aus:
                    best_aus = aus_value
                    best_model_state = deepcopy(self.net.state_dict())
                    best_epoch = epoch

                    best_acc_train_ret = acc_train_ret
                    best_acc_train_fgt = acc_train_fgt
                    best_acc_test_val_ret = acc_test_val_ret
                    best_acc_test_val_fgt = acc_test_val_fgt
                    best_acc_full_val_ret = acc_full_val_ret
                    best_acc_full_val_fgt = acc_full_val_fgt

                if acc_test_val_fgt == 0.0:
                    zero_acc_fgt_counter += 1
                else:
                    zero_acc_fgt_counter = 0
                if zero_acc_fgt_counter >= zero_acc_patience:
                    print(f"[Early Stopping] Forget accuracy 0 for {zero_acc_patience} epochs.")
                    break
                if len(aus_history) > opt.patience:
                    recent_trend = aus_history[-opt.patience:]
                    if all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)) or all(abs(recent_trend[i] - recent_trend[i+1]) < 1e-4 for i in range(len(recent_trend)-1)):
                        print(f"[Early Stopping] AUS decreased or flat for {opt.patience} epochs.")
                        break
                if sum(a < 0.4 for a in aus_history[-20:]) >= 20:
                    print(f"[Early Stopping] AUS < 0.4 for 20 epochs.")
                    break

                log_epoch_to_csv(
                    epoch=epoch,
                    epoch_times=duration,
                    train_retain_acc=round(acc_train_ret, 4),
                    train_fgt_acc=round(acc_train_fgt, 4),
                    val_test_retain_acc=round(acc_test_val_ret, 4),
                    val_test_fgt_acc=round(acc_test_val_fgt, 4),
                    val_full_retain_acc=round(acc_full_val_ret, 4),
                    val_full_fgt_acc=round(acc_full_val_fgt, 4),
                    AUS=round(aus_value, 4),
                    mode=opt.method,
                    dataset=opt.dataset,
                    model=opt.model,
                    class_to_remove=self.class_to_remove,
                    seed=opt.seed,
                    retain_count=retain_count,
                    forget_count=forget_count,
                    total_count=total_count)
                
                
            self.scheduler.step()

        unlearning_time_until_best = sum(epoch_times[:best_epoch + 1])

        log_summary_across_classes(
            best_epoch=best_epoch,
            train_retain_acc=round(best_acc_train_ret, 4),
            train_fgt_acc=round(best_acc_train_fgt, 4),
            val_test_retain_acc=round(best_acc_test_val_ret, 4),
            val_test_fgt_acc=round(best_acc_test_val_fgt, 4),
            val_full_retain_acc=round(best_acc_full_val_ret, 4),
            val_full_fgt_acc=round(best_acc_full_val_fgt, 4),
            AUS=round(best_aus, 4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best,4))
        
        
        self.net.load_state_dict(best_model_state)
        self.net.eval()
        return self.net


class BadTeacher(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader,
                 retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader,
                         test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        self.class_to_remove = class_to_remove
        self.loader = self._create_combined_loader()
        self.full_teacher = deepcopy(self.net).eval()
        self.unlearning_teacher = deepcopy(self.net).eval()
        torch.nn.init.normal_(self.unlearning_teacher.fc.weight, mean=0.0, std=0.02)
        if self.unlearning_teacher.fc.bias is not None:
            torch.nn.init.constant_(self.unlearning_teacher.fc.bias, 0)
        self.unlearning_teacher.to(opt.device).eval()
        self.KL_temperature = 1.0
        self.head_fc = get_classifier(self.net)

    def _create_combined_loader(self):
        class UnLearnDataset(torch.utils.data.Dataset):
            def __init__(self, forget_data, retain_data):
                self.forget_data = forget_data
                self.retain_data = retain_data
                self.forget_len = len(forget_data)
                self.retain_len = len(retain_data)

            def __len__(self):
                return self.forget_len + self.retain_len

            def __getitem__(self, index):
                if index < self.forget_len:
                    x, y = self.forget_data[index]
                    return x.detach().cpu(), y.detach().cpu(), torch.tensor(1, dtype=torch.long)
                else:
                    x, y = self.retain_data[index - self.forget_len]
                    return x.detach().cpu(), y.detach().cpu(), torch.tensor(0, dtype=torch.long)

        combined_dataset = UnLearnDataset(
            self.train_fgt_loader.dataset, self.train_retain_loader.dataset)

        return DataLoader(combined_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)


    def loss_f(self, inputs, targets):
        raise NotImplementedError("Unused in BadTeacherCustom")

    def run(self):
        best_model_state = None
        best_aus = -float('inf')
        best_epoch = -1
        optimizer = torch.optim.SGD(self.net.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        a_or_value = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        epoch_times = []

        def distill_loss(student_logits, unlearn_labels, full_teacher_logits, unlearning_teacher_logits):
            unlearn_labels = unlearn_labels.unsqueeze(1).float()
            f_teacher_out = F.softmax(full_teacher_logits / self.KL_temperature, dim=1)
            u_teacher_out = F.softmax(unlearning_teacher_logits / self.KL_temperature, dim=1)
            overall_teacher_out = unlearn_labels * u_teacher_out + (1 - unlearn_labels) * f_teacher_out
            student_out = F.log_softmax(student_logits / self.KL_temperature, dim=1)
            return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            self.net.train()
            for x, y, unlearn_label in self.loader:
                x, y, unlearn_label = x.to(opt.device), y.to(opt.device), unlearn_label.to(opt.device)
                with torch.no_grad():
                    full_teacher_logits = self.full_teacher.fc(x)
                    unlearning_teacher_logits = self.unlearning_teacher.fc(x)
                student_logits = self.net.fc(x)
                loss = distill_loss(student_logits, unlearn_label, full_teacher_logits, unlearning_teacher_logits)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)

            with torch.no_grad():
                self.net.eval()
                acc_train_ret = calculate_accuracy(self.net, self.train_retain_loader, use_fc_only=True)
                acc_train_fgt = calculate_accuracy(self.net, self.train_fgt_loader, use_fc_only=True)
                acc_test_val_ret = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
                acc_test_val_fgt = calculate_accuracy(self.net, self.test_fgt_loader, use_fc_only=True)
                acc_full_val_ret = calculate_accuracy(self.net, self.retainfull_loader_real, use_fc_only=True)
                acc_full_val_fgt = calculate_accuracy(self.net, self.forgetfull_loader_real, use_fc_only=True)

                a_t = Complex(acc_test_val_ret, 0.0)
                a_f = Complex(acc_test_val_fgt, 0.0)
                a_or = Complex(a_or_value, 0.0)
                aus_value = AUS(a_t, a_or, a_f).value

                if aus_value > best_aus:
                    best_aus = aus_value
                    best_model_state = deepcopy(self.net.state_dict())
                    best_epoch = epoch
                    best_accs = (acc_train_ret, acc_train_fgt, acc_test_val_ret, acc_test_val_fgt, acc_full_val_ret, acc_full_val_fgt)

                    checkpoint_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt")
                    torch.save(best_model_state, checkpoint_path)
                    print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={aus_value:.4f} to {checkpoint_path}")


                print(f"Epoch {epoch+1}/{self.epochs} | "
                        f"Train Retain Acc: {acc_train_ret:.2f} | "
                        f"Train Forget Acc: {acc_train_fgt:.2f} | "
                        f"Val Retain Test Acc: {acc_test_val_ret:.2f} | "
                        f"Val Forget Test Acc: {acc_test_val_fgt:.2f} | "
                        f"Val Retain Full Acc: {acc_full_val_ret:.2f} | "
                        f"Val Forget Full Acc: {acc_full_val_fgt:.2f} | "
                        f"AUS: {aus_value:.4f}")



                log_epoch_to_csv(epoch,
                                 duration,
                                 acc_train_ret,
                                 acc_train_fgt,
                                 acc_test_val_ret,
                                 acc_test_val_fgt,
                                 acc_full_val_ret,
                                 acc_full_val_fgt,
                                 aus_value,
                                 opt.method,
                                 opt.dataset,
                                 opt.model,
                                 self.class_to_remove,
                                 opt.seed,
                                 retain_count,
                                 forget_count,
                                 total_count)

            scheduler.step()

        unlearning_time = sum(epoch_times[:best_epoch + 1])
        best_accs = tuple(round(a, 4) for a in best_accs)
        log_summary_across_classes(best_epoch, *best_accs, round(best_aus, 4), opt.method, opt.dataset, opt.model,
                                   self.class_to_remove, opt.seed, retain_count, forget_count, total_count, round(unlearning_time, 4))
        self.net.load_state_dict(best_model_state)
        return self.net


class Delete(BaseMethod):
    """
    FC-only delete:
      - Freeze backbone; train only net.fc
      - Teacher = frozen deepcopy of initial net (FC included)
      - On forget samples, set teacher logit[target] ≈ -∞ (anti-target),
        train student FC via KLDiv (log_softmax vs softmax) to match it.
    """
    def __init__(self, net,
                 train_retain_loader, train_fgt_loader,
                 test_retain_loader,  test_fgt_loader,
                 retainfull_loader_real, forgetfull_loader_real,
                 class_to_remove=None):
        super().__init__(net,
                         train_retain_loader, train_fgt_loader,
                         test_retain_loader,  test_fgt_loader,
                         retainfull_loader_real, forgetfull_loader_real)
        self.class_to_remove = class_to_remove

        # 0) Freeze policy: FC-only training
        freeze_all_but_fc(self.net)

        # 1) Frozen teacher
        self.teacher = deepcopy(self.net).to(opt.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 2) Loss / optimizer / scheduler
        self.kl = nn.KLDivLoss(reduction='batchmean')
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Delete: no trainable params (expected FC)."

        self.optimizer = optim.SGD(
            # keep trainable if you're doing FC-only; use self.net.parameters() to match "all params"
            get_classifier(self.net).parameters(),
            lr=opt.lr_unlearn,
            momentum=0.9,          # match original
            weight_decay=0.0,      # original had none
            nesterov=False         # original did not enable it
        )

        # Use your scheme or cosine—either is fine. Keeping cosine here:
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # 3) Baseline retain accuracy for AUS
        self.a_or_value = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)

        # 4) Early-stop bookkeeping
        self.zero_acc_patience = 1000
        self.patience = opt.patience
        self.head_fc = get_classifier(self.net)


    def _teacher_probs_anti(self, x, y):
        with torch.no_grad():
            head_t   = get_classifier(self.teacher)
            t_logits = head_t(x)
            # avoid in-place on a tensor you didn’t create
            t_logits = t_logits.clone()
            t_logits.scatter_(1, y.view(-1, 1), -1e9)
            probs = F.softmax(t_logits, dim=1)
        return probs # a normal (non-inference) tensor OK for KL target


    def run(self):
        self.net.train()
        best_state = None
        best_aus   = -float('inf')
        best_epoch = -1
        aus_hist   = []
        zero_fgt   = 0
        epoch_times = []

        # Initialize best_* so summary never references uninitialized vars
        with torch.no_grad():
            self.net.eval()
            init_train_ret    = calculate_accuracy(self.net, self.train_retain_loader, use_fc_only=True)
            init_train_fgt    = calculate_accuracy(self.net, self.train_fgt_loader,   use_fc_only=True)
            init_test_val_ret = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
            init_test_val_fgt = calculate_accuracy(self.net, self.test_fgt_loader,    use_fc_only=True)
            init_full_val_ret = calculate_accuracy(self.net, self.retainfull_loader_real, use_fc_only=True)
            init_full_val_fgt = calculate_accuracy(self.net, self.forgetfull_loader_real, use_fc_only=True)
            self.net.train()

        best_acc_train_ret    = init_train_ret
        best_acc_train_fgt    = init_train_fgt
        best_acc_test_val_ret = init_test_val_ret
        best_acc_test_val_fgt = init_test_val_fgt
        best_acc_full_val_ret = init_full_val_ret
        best_acc_full_val_fgt = init_full_val_fgt

        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count  = retain_count + forget_count

        for epoch in tqdm(range(self.epochs)):
            t0 = time.time()
            for xf, yf in self.train_fgt_loader:
                xf, yf = xf.to(opt.device), yf.to(opt.device)

                tgt_prob = self._teacher_probs_anti(xf, yf)             
                head_s = get_classifier(self.net)
                logits = head_s(xf)                       
                loss     = self.kl(F.log_softmax(logits, dim=1), tgt_prob)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            epoch_times.append(time.time() - t0)

            # ---- EVAL + AUS ----
            with torch.no_grad():
                self.net.eval()
                acc_train_ret     = calculate_accuracy(self.net, self.train_retain_loader, use_fc_only=True)
                acc_train_fgt     = calculate_accuracy(self.net, self.train_fgt_loader,   use_fc_only=True)
                acc_test_val_ret  = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True)
                acc_test_val_fgt  = calculate_accuracy(self.net, self.test_fgt_loader,    use_fc_only=True)
                acc_full_val_ret  = calculate_accuracy(self.net, self.retainfull_loader_real, use_fc_only=True)
                acc_full_val_fgt  = calculate_accuracy(self.net, self.forgetfull_loader_real, use_fc_only=True)
                self.net.train()

                a_t  = Complex(acc_test_val_ret, 0.0)
                a_f  = Complex(acc_test_val_fgt, 0.0)
                a_or = Complex(self.a_or_value,  0.0)
                aus  = AUS(a_t, a_or, a_f).value
                aus_hist.append(aus)

                print(f"[epoch {epoch}] loss={loss.item():.4f} | "
                      f"Train Ret {acc_train_ret:.3f}  Fgt {acc_train_fgt:.3f} | "
                      f"Val Ret {acc_test_val_ret:.3f}  Fgt {acc_test_val_fgt:.3f} | "
                      f"Full Ret {acc_full_val_ret:.3f}  Fgt {acc_full_val_fgt:.3f} | AUS {aus:.3f}")

                # checkpoint on best AUS
                if aus > best_aus:
                    best_aus   = aus
                    best_epoch = epoch
                    best_state = deepcopy(self.net.state_dict())
                    best_acc_train_ret    = acc_train_ret
                    best_acc_train_fgt    = acc_train_fgt
                    best_acc_test_val_ret = acc_test_val_ret
                    best_acc_test_val_fgt = acc_test_val_fgt
                    best_acc_full_val_ret = acc_full_val_ret
                    best_acc_full_val_fgt = acc_full_val_fgt

                if getattr(opt, "save_model", False):  # <-- only save if enabled
                    ckpt_dir = f"checkpoints_main/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(
                        ckpt_dir,
                        f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                    )
                    torch.save(best_state, ckpt_path)
                    print(f"[Checkpoint Saved] AUS={aus:.4f} -> {ckpt_path}")
                else:
                    print(f"[Checkpoint NOT saved] AUS={aus:.4f} (kept in memory)")

                # early-stops (like your NGFT_weighted)
                if acc_test_val_fgt == 0.0:
                    zero_fgt += 1
                else:
                    zero_fgt = 0
                if zero_fgt >= self.zero_acc_patience:
                    print(f"[Early Stopping] forget acc=0 for {self.zero_acc_patience} consecutive epochs.")
                    break

                if len(aus_hist) > self.patience:
                    recent = aus_hist[-self.patience:]
                    decreasing = all(recent[i] > recent[i+1] for i in range(self.patience-1))
                    no_change  = all(abs(recent[i] - recent[i+1]) < 1e-4 for i in range(self.patience-1))
                    if decreasing or no_change:
                        print(f"[Early Stopping] AUS trend at epoch {epoch+1}.")
                        break

                if len(aus_hist) >= 20 and sum(a < 0.4 for a in aus_hist[-20:]) >= 20:
                    print(f"[Early Stopping] AUS < 0.4 for 20 consecutive epochs.")
                    break

                log_epoch_to_csv(
                    epoch=epoch,
                    epoch_times=epoch_times[-1],
                    train_retain_acc=round(acc_train_ret, 4),
                    train_fgt_acc=round(acc_train_fgt, 4),
                    val_test_retain_acc=round(acc_test_val_ret, 4),
                    val_test_fgt_acc=round(acc_test_val_fgt, 4),
                    val_full_retain_acc=round(acc_full_val_ret, 4),
                    val_full_fgt_acc=round(acc_full_val_fgt, 4),
                    AUS=round(aus, 4),
                    mode=opt.method,
                    dataset=opt.dataset,
                    model=opt.model,
                    class_to_remove=self.class_to_remove,
                    seed=opt.seed,
                    retain_count=retain_count,
                    forget_count=forget_count,
                    total_count=total_count
                )

            self.scheduler.step()

        # restore best FC
        if best_state is not None:
            self.net.load_state_dict(best_state)

        # summary CSV
        unlearning_time_until_best = sum(epoch_times[:best_epoch + 1]) if best_epoch >= 0 else 0.0
        log_summary_across_classes(
            best_epoch=best_epoch,
            train_retain_acc=round(best_acc_train_ret, 4),
            train_fgt_acc=round(best_acc_train_fgt, 4),
            val_test_retain_acc=round(best_acc_test_val_ret, 4),
            val_test_fgt_acc=round(best_acc_test_val_fgt, 4),
            val_full_retain_acc=round(best_acc_full_val_ret, 4),
            val_full_fgt_acc=round(best_acc_full_val_fgt, 4),
            AUS=round(best_aus, 4),
            mode=opt.method,
            dataset=opt.dataset,
            model=opt.model,
            class_to_remove=self.class_to_remove,
            seed=opt.seed,
            retain_count=retain_count,
            forget_count=forget_count,
            total_count=total_count,
            unlearning_time_until_best=round(unlearning_time_until_best, 4)
        )

        self.net.eval()
        return self.net
