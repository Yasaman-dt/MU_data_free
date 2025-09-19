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
import copy

from generate_part_samples_randomly_resnet18 import TruncatedResNet, RemainingResNet





n_model = opt.n_model
    
def AUS(a_t, a_or, a_f):
    aus=(Complex(1, 0)-(a_or-a_t))/(Complex(1, 0)+abs(a_f))
    return aus

def choose_method(name):
    if name=='FT':
        return FineTuning
    elif name=='NG':
        return NegativeGradient
    elif name=='NGFTW':
        return NGFT_weighted
    elif name=='RL':
        return RandomLabels
    else:
        raise ValueError(f"[choose_method] Unknown method: {name}")
    
    
def count_samples(dataloader):
    return sum(inputs.size(0) for inputs, _ in dataloader)
        
def calculate_accuracy(net, dataloader, use_fc_only=False):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            if use_fc_only:
                outputs = net.fc(inputs)  # Only use the FC layer
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

    return 100 * correct / total if total > 0 else 0






def log_epoch_to_csv(epoch, epoch_times,train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, mode, dataset, model, class_to_remove, seed, retain_count, forget_count,total_count):
    os.makedirs(f'results_synth/samples_per_class_{opt.samples_per_class}/{mode}/epoch_logs_m{n_model}_lr{opt.lr_unlearn}', exist_ok=True)

    if isinstance(class_to_remove, list):
        class_name = '_'.join(map(str, class_to_remove))
    else:
        class_name = class_to_remove if class_to_remove is not None else 'all'

    csv_path = f'results_synth/samples_per_class_{opt.samples_per_class}/{mode}/epoch_logs_m{n_model}_lr{opt.lr_unlearn}/{dataset}_{model}_epoch_results_m{n_model}_{class_name}.csv'
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch', 'epoch_times', 'mode', 'Forget Class', 'seed', 'train_retain_acc', 'train_fgt_acc', 'val_test_retain_acc', 'val_test_fgt_acc', 'val_full_retain_acc', 'val_full_fgt_acc', 'AUS', 'retain_count', 'forget_count','total_count'])
        writer.writerow([epoch, epoch_times, mode, class_name, seed, train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, retain_count, forget_count,total_count])

def log_summary_across_classes(best_epoch, train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, mode, dataset, model, class_to_remove, seed, retain_count, forget_count,total_count, unlearning_time_until_best):
    os.makedirs('results_synth', exist_ok=True)
    summary_path = f'results_synth/samples_per_class_{opt.samples_per_class}/{mode}/{dataset}_{model}_unlearning_summary_m{n_model}_lr{opt.lr_unlearn}.csv'
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

        
class BaseMethod:
    def __init__(self,
                 net,
                 train_retain_loader_img,
                 train_fgt_loader_img,
                 test_retain_loader_img,
                 test_fgt_loader_img,
                 train_retain_loader,
                 train_fgt_loader,
                 test_retain_loader,
                 test_fgt_loader,
                 retainfull_loader_real,
                 forgetfull_loader_real,
                 class_to_remove=None):
        
        self.net = net
        self.train_retain_loader = train_retain_loader
        self.train_fgt_loader = train_fgt_loader
        self.test_retain_loader = test_retain_loader
        self.test_fgt_loader = test_fgt_loader
        self.retainfull_loader_real = retainfull_loader_real
        self.forgetfull_loader_real = forgetfull_loader_real
        self.train_retain_loader_img = train_retain_loader_img
        self.train_fgt_loader_img = train_fgt_loader_img
        self.test_retain_loader_img = test_retain_loader_img
        self.test_fgt_loader_img = test_fgt_loader_img

        self.class_to_remove = class_to_remove

        self.Truncatedmodel = TruncatedResNet(self.net).to(opt.device)
        self.Remainingmodel = RemainingResNet(self.net).to(opt.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.Remainingmodel.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
        self.epochs = opt.epochs_unlearn
        self.target_accuracy = opt.target_accuracy
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.scheduler, gamma=0.5)

        for images, labels in self.train_retain_loader:
            print("train_retain_loader_synth:", images.shape, labels.shape)
            break

        for images, labels in self.train_fgt_loader:
            print("train_fgt_loader_synth:", images.shape, labels.shape)
            break

        for images, labels in self.test_retain_loader_img:
            print("test_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_fgt_loader_img:
            print("test_fgt_loader_img:", images.shape, labels.shape)
            break
        
        print(f"Train Retain Samples: {len(self.train_retain_loader.dataset)}")
        print(f"Test Retain Samples: {len(self.test_retain_loader.dataset)}")
        print(f"Real Retain Full Samples: {len(self.retainfull_loader_real.dataset)}")
        print(f"Train forget Samples: {len(self.train_fgt_loader.dataset)}")
        print(f"Test forget Samples: {len(self.test_fgt_loader.dataset)}")
        print(f"Real forget Full Samples: {len(self.forgetfull_loader_real.dataset)}")

    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        for param in self.Truncatedmodel.parameters():
            param.requires_grad = False
            
        for param in self.Remainingmodel.parameters():
            param.requires_grad = True        
        
        self.Remainingmodel.train()
        best_model_state = None
        best_aus = -float('inf')
        best_epoch = -1
        patience_counter = 0
        patience = opt.patience



        merged_model = copy.deepcopy(self.net)

        # === Overwrite the parts that were trained in RemainingResNet ===
        merged_model.layer4[1].conv2 = self.Remainingmodel.layer4_1_conv2
        merged_model.layer4[1].bn2 = self.Remainingmodel.layer4_1_bn2
        merged_model.avgpool = self.Remainingmodel.avgpool
        merged_model.fc = self.Remainingmodel.fc

        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 1000    # Stop if this happens for 50+ consecutive epochs

        aus_history = []
        results = []
        a_or_value = evaluate_embedding_accuracy(merged_model, self.test_retain_loader_img, opt.device)/100
        print(a_or_value)
        
        a_or_value = evaluate_embedding_accuracy(merged_model.fc, self.test_retain_loader, opt.device)/100
        print(a_or_value)
        
        retain_count = count_samples(self.train_retain_loader_img)
        forget_count = count_samples(self.train_fgt_loader_img)
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
                acc_train_ret = evaluate_embedding_accuracy(self.Remainingmodel, self.train_retain_loader, opt.device)/100
                acc_train_fgt = evaluate_embedding_accuracy(self.Remainingmodel, self.train_fgt_loader, opt.device)/100
                acc_test_val_ret = evaluate_embedding_accuracy(merged_model, self.test_retain_loader_img, opt.device)/100
                acc_test_val_fgt = evaluate_embedding_accuracy(merged_model, self.test_fgt_loader_img, opt.device)/100
                acc_full_val_ret = evaluate_embedding_accuracy(merged_model.fc, self.retainfull_loader_real, opt.device)/100
                acc_full_val_fgt = evaluate_embedding_accuracy(merged_model.fc, self.forgetfull_loader_real, opt.device)/100



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
                    best_model_state = deepcopy(merged_model.state_dict())
                    best_epoch = epoch

                    best_acc_train_ret = acc_train_ret
                    best_acc_train_fgt = acc_train_fgt
                    best_acc_test_val_ret = acc_test_val_ret
                    best_acc_test_val_fgt = acc_test_val_fgt
                    best_acc_full_val_ret = acc_full_val_ret
                    best_acc_full_val_fgt = acc_full_val_fgt


                    # checkpoint_dir = f"checkpoints_main_part/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                    # os.makedirs(checkpoint_dir, exist_ok=True)

                    # checkpoint_path = os.path.join(
                    #     checkpoint_dir,
                    #     f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                    # )

                    # torch.save(best_model_state, checkpoint_path)
                    # print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={aus_value:.4f} to {checkpoint_path}")



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

        merged_model.eval()
        return merged_model
    
    
class FineTuning(BaseMethod):
    def __init__(self,
                 net,
                 train_retain_loader_img,
                 train_fgt_loader_img,
                 test_retain_loader_img,
                 test_fgt_loader_img,
                 train_retain_loader,
                 train_fgt_loader,
                 test_retain_loader,
                 test_fgt_loader,
                 retainfull_loader_real,
                 forgetfull_loader_real,
                 class_to_remove=None):
        
        super().__init__(net,
                         train_retain_loader_img,
                         train_fgt_loader_img,
                         test_retain_loader_img,
                         test_fgt_loader_img,
                         train_retain_loader,
                         train_fgt_loader,
                         test_retain_loader,
                         test_fgt_loader,
                         retainfull_loader_real,
                         forgetfull_loader_real)
        
        self.loader = self.train_retain_loader
        self.target_accuracy=0.0
        self.class_to_remove = class_to_remove

    def loss_f(self, inputs_r, targets_r ,test=None):
        loss = self.criterion(self.Remainingmodel(inputs_r), targets_r)
        return loss

class RandomLabels(BaseMethod):
    def __init__(self,
                 net,
                 train_retain_loader_img,
                 train_fgt_loader_img,
                 test_retain_loader_img,
                 test_fgt_loader_img,
                 train_retain_loader,
                 train_fgt_loader,
                 test_retain_loader,
                 test_fgt_loader,
                 retainfull_loader_real,
                 forgetfull_loader_real,
                 class_to_remove=None):
        super().__init__(net,
                         train_retain_loader_img,
                         train_fgt_loader_img,
                         test_retain_loader_img,
                         test_fgt_loader_img,
                         train_retain_loader,
                         train_fgt_loader,
                         test_retain_loader,
                         test_fgt_loader,
                         retainfull_loader_real,
                         forgetfull_loader_real)
        self.loader = self.train_fgt_loader
        self.class_to_remove = class_to_remove
        self.Truncatedmodel = TruncatedResNet(self.net).to(opt.device)
        self.Remainingmodel = RemainingResNet(self.net).to(opt.device)
        
        if opt.mode == "CR":
            self.random_possible = torch.tensor([i for i in range(opt.num_classes) if i not in self.class_to_remove]).to(opt.device).to(torch.float32)

    def loss_f(self, inputs, targets):
        outputs = self.Remainingmodel(inputs)
        #create a random label tensor of the same shape as the outputs chosing values from self.possible_labels
        random_labels = self.random_possible[torch.randint(low=0, high=self.random_possible.shape[0], size=targets.shape)].to(torch.int64).to(opt.device)
        loss = self.criterion(outputs, random_labels)
        return loss

class NegativeGradient(BaseMethod):
    def __init__(self,
                 net,
                 train_retain_loader_img,
                 train_fgt_loader_img,
                 test_retain_loader_img,
                 test_fgt_loader_img,
                 train_retain_loader,
                 train_fgt_loader,
                 test_retain_loader,
                 test_fgt_loader,
                 retainfull_loader_real,
                 forgetfull_loader_real,
                 class_to_remove=None):
        super().__init__(net,
                         train_retain_loader_img,
                         train_fgt_loader_img,
                         test_retain_loader_img,
                         test_fgt_loader_img,
                         train_retain_loader,
                         train_fgt_loader,
                         test_retain_loader,
                         test_fgt_loader,
                         retainfull_loader_real,
                         forgetfull_loader_real)
        
        self.Truncatedmodel = TruncatedResNet(self.net).to(opt.device)
        self.Remainingmodel = RemainingResNet(self.net).to(opt.device)
        self.loader = self.train_fgt_loader
        self.class_to_remove = class_to_remove


        for images, labels in self.train_retain_loader:
            print("train_retain_loader_synth:", images.shape, labels.shape)
            break

        for images, labels in self.train_fgt_loader:
            print("train_fgt_loader_synth:", images.shape, labels.shape)
            break

        for images, labels in self.test_retain_loader_img:
            print("test_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_fgt_loader_img:
            print("test_fgt_loader_img:", images.shape, labels.shape)
            break


    def loss_f(self, inputs_f, targets_f):
        loss = self.criterion(self.Remainingmodel(inputs_f), targets_f) * (-1)
        return loss
 

class NGFT_weighted(BaseMethod):
    def __init__(self,
                 net,
                 train_retain_loader_img,
                 train_fgt_loader_img,
                 test_retain_loader_img,
                 test_fgt_loader_img,
                 train_retain_loader,
                 train_fgt_loader,
                 test_retain_loader,
                 test_fgt_loader,
                 retainfull_loader_real,
                 forgetfull_loader_real,
                 class_to_remove=None):
        
        super().__init__(net,
                         train_retain_loader_img,
                         train_fgt_loader_img,
                         test_retain_loader_img,
                         test_fgt_loader_img,
                         train_retain_loader,
                         train_fgt_loader,
                         test_retain_loader,
                         test_fgt_loader,
                         retainfull_loader_real,
                         forgetfull_loader_real)


                                                     
        self.train_retain_loader_img = train_retain_loader_img
        self.train_fgt_loader_img = train_fgt_loader_img
        self.test_retain_loader_img = test_retain_loader_img
        self.test_fgt_loader_img = test_fgt_loader_img
        
        
        self.train_retain_loader = train_retain_loader
        self.train_fgt_loader = train_fgt_loader
        self.test_retain_loader = test_retain_loader
        self.test_fgt_loader = test_fgt_loader
        self.retainfull_loader_real = retainfull_loader_real
        self.forgetfull_loader_real = forgetfull_loader_real
        self.class_to_remove = class_to_remove
        self.beta = 0.9  # balance factor
        self.Truncatedmodel = TruncatedResNet(self.net).to(opt.device)
        self.Remainingmodel = RemainingResNet(self.net).to(opt.device)
        self.optimizer = torch.optim.Adam(self.Remainingmodel.parameters(), lr=opt.lr_unlearn)

        for images, labels in self.train_retain_loader:
            print("train_retain_loader_synth:", images.shape, labels.shape)
            break

        for images, labels in self.train_fgt_loader:
            print("train_fgt_loader_synth:", images.shape, labels.shape)
            break

        for images, labels in self.test_retain_loader_img:
            print("test_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_fgt_loader_img:
            print("test_fgt_loader_img:", images.shape, labels.shape)
            break
        
        
        
        
    def loss_weighted(self, inputs_r, targets_r, inputs_f, targets_f):
        retain_loss = self.criterion(self.Remainingmodel(inputs_r), targets_r)
        forget_loss = self.criterion(self.Remainingmodel(inputs_f), targets_f)
        # Weighted sum as per the formula
        return self.beta * retain_loss - (1 - self.beta) * forget_loss

    def run(self):
        for param in self.Truncatedmodel.parameters():
            param.requires_grad = False
            
        for param in self.Remainingmodel.parameters():
            param.requires_grad = True
            
        self.Remainingmodel.train()
        best_model_state = None
        best_aus = -float('inf')
        best_epoch = -1
        aus_history = []
        zero_acc_fgt_counter = 0
        zero_acc_patience = 1000
        patience = opt.patience


        merged_model = copy.deepcopy(self.net)

        # === Overwrite the parts that were trained in RemainingResNet ===
        merged_model.layer4[1].conv2 = self.Remainingmodel.layer4_1_conv2
        merged_model.layer4[1].bn2 = self.Remainingmodel.layer4_1_bn2
        merged_model.avgpool = self.Remainingmodel.avgpool
        merged_model.fc = self.Remainingmodel.fc

        a_or_value = evaluate_embedding_accuracy(merged_model, self.test_retain_loader_img, opt.device)/100
        print(a_or_value)
        
        a_or_value = evaluate_embedding_accuracy(merged_model.fc, self.test_retain_loader, opt.device)/100
        print(a_or_value)
        
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


            with torch.no_grad():
                self.net.eval()
                acc_train_ret = evaluate_embedding_accuracy(self.Remainingmodel, self.train_retain_loader, opt.device)/100
                acc_train_fgt = evaluate_embedding_accuracy(self.Remainingmodel, self.train_fgt_loader, opt.device)/100
                acc_test_val_ret = evaluate_embedding_accuracy(merged_model, self.test_retain_loader_img, opt.device)/100
                acc_test_val_fgt = evaluate_embedding_accuracy(merged_model, self.test_fgt_loader_img, opt.device)/100
                acc_full_val_ret = evaluate_embedding_accuracy(merged_model.fc, self.retainfull_loader_real, opt.device)/100
                acc_full_val_fgt = evaluate_embedding_accuracy(merged_model.fc, self.forgetfull_loader_real, opt.device)/100
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
                    best_model_state = deepcopy(merged_model.state_dict())
                    best_epoch = epoch
                    best_acc_train_ret = acc_train_ret
                    best_acc_train_fgt = acc_train_fgt
                    best_acc_test_val_ret = acc_test_val_ret
                    best_acc_test_val_fgt = acc_test_val_fgt
                    best_acc_full_val_ret = acc_full_val_ret
                    best_acc_full_val_fgt = acc_full_val_fgt


                    # checkpoint_dir = f"checkpoints_main_part/{opt.dataset}/{opt.method}/samples_per_class_{opt.samples_per_class}"
                    # os.makedirs(checkpoint_dir, exist_ok=True)

                    # checkpoint_path = os.path.join(
                    #     checkpoint_dir,
                    #     f"{opt.model}_best_checkpoint_seed{opt.seed}_class{self.class_to_remove}_m{n_model}_lr{opt.lr_unlearn}.pt"
                    # )

                    # torch.save(best_model_state, checkpoint_path)
                    # print(f"[Checkpoint Saved] Best model saved at epoch {epoch} with AUS={aus_value:.4f} to {checkpoint_path}")







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




        merged_model.eval()
        return merged_model

