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
from random import choice
import copy

from generate_part_samples_randomly import TruncatedResNet, RemainingResNet



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
    elif name == 'newmethod':
        return newmethod
    elif name == 'RE':
        return RetrainedEmbedding
    elif name == 'MM':
        return MaximeMethod
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
def log_epoch_to_csv(epoch, epoch_times, train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, mode, dataset, model, class_to_remove, seed, retain_count, forget_count,total_count):
    os.makedirs(f'results_real/samples_per_class_{opt.samples_per_class}/{mode}/epoch_logs_m{n_model}_lr{opt.lr_unlearn}', exist_ok=True)

    if isinstance(class_to_remove, list):
        class_name = '_'.join(map(str, class_to_remove))
    else:
        class_name = class_to_remove if class_to_remove is not None else 'all'

    csv_path = f'results_real/samples_per_class_{opt.samples_per_class}/{mode}/epoch_logs_m{n_model}_lr{opt.lr_unlearn}/{dataset}_{model}_epoch_results_m{n_model}_{class_name}.csv'
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['epoch','epoch_times',  'mode', 'Forget Class', 'seed', 'train_retain_acc', 'train_fgt_acc', 'val_test_retain_acc', 'val_test_fgt_acc', 'val_full_retain_acc', 'val_full_fgt_acc', 'AUS', 'retain_count', 'forget_count','total_count'])
        writer.writerow([epoch, epoch_times, mode, class_name, seed, train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, retain_count, forget_count,total_count])

def log_summary_across_classes(best_epoch, train_retain_acc, train_fgt_acc, val_test_retain_acc, val_test_fgt_acc, val_full_retain_acc, val_full_fgt_acc, AUS, mode, dataset, model, class_to_remove, seed, retain_count, forget_count,total_count, unlearning_time_until_best):
    os.makedirs('results_real', exist_ok=True)
    summary_path = f'results_real/samples_per_class_{opt.samples_per_class}/{mode}/{dataset}_{model}_unlearning_summary_m{n_model}_lr{opt.lr_unlearn}.csv'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
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


        for images, labels in self.train_retain_loader_img:
            print("train_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.train_fgt_loader_img:
            print("train_fgt_loader_img:", images.shape, labels.shape)
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
        
    def loss_f(self, merged_model, inputs, targets):
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
        a_or_value = evaluate_embedding_accuracy(merged_model, self.test_retain_loader_img, opt.device)/100
        print(a_or_value)
        
        a_or_value = evaluate_embedding_accuracy(merged_model.fc, self.test_retain_loader, opt.device)/100
        print(a_or_value)
        
        zero_acc_fgt_counter = 0  # Track consecutive epochs with acc_test_fgt == 0
        zero_acc_patience = 50    # Stop if this happens for 50+ consecutive epochs
        aus_history = []
        results = []
        epoch_times = []

        retain_count = count_samples(self.train_retain_loader_img)
        forget_count = count_samples(self.train_fgt_loader_img)
        total_count = retain_count + forget_count
        
        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()

            for inputs, targets in self.loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(merged_model, inputs, targets)
                loss.backward()
                self.optimizer.step()

            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)
            with torch.no_grad():
                self.net.eval()
                acc_train_ret = evaluate_embedding_accuracy(merged_model, self.train_retain_loader_img, opt.device)/100
                acc_train_fgt = evaluate_embedding_accuracy(merged_model, self.train_fgt_loader_img, opt.device)/100
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
            #print('Accuracy: ',self.evalNet())

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
    
    def evalNet(self):
        #compute model accuracy on self.loader

        self.net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in self.train_retain_loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            correct2 = 0
            total2 = 0
            for inputs, targets in self.train_fgt_loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total2 += targets.size(0)
                correct2+= (predicted == targets).sum().item()

            if not(self.test is None):
                correct3 = 0
                total3 = 0
                for inputs, targets in self.test:
                    inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                    outputs = self.net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total3 += targets.size(0)
                    correct3+= (predicted == targets).sum().item()
        self.net.train()
        if self.test is None:
            return correct/total,correct2/total2
        else:
            return correct/total,correct2/total2,correct3/total3
    
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
        
        self.loader = self.train_retain_loader_img
        self.target_accuracy=0.0
        self.class_to_remove = class_to_remove
        self.Truncatedmodel = TruncatedResNet(self.net).to(opt.device)
        self.Remainingmodel = RemainingResNet(self.net).to(opt.device)
        merged_model = copy.deepcopy(self.net)
        
    def loss_f(self, merged_model, inputs_r, targets_r):
        loss = self.criterion(merged_model(inputs_r), targets_r)
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
        
        self.loader = self.train_fgt_loader_img
        self.class_to_remove = class_to_remove
        self.Truncatedmodel = TruncatedResNet(self.net).to(opt.device)
        self.Remainingmodel = RemainingResNet(self.net).to(opt.device)
        merged_model = copy.deepcopy(self.net)

        
        
        
        if opt.mode == "CR":
            self.random_possible = torch.tensor([i for i in range(opt.num_classes) if i not in self.class_to_remove]).to(opt.device).to(torch.float32)


       
        
        for images, labels in self.train_retain_loader_img:
            print("train_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.train_fgt_loader_img:
            print("train_fgt_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_retain_loader_img:
            print("test_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_fgt_loader_img:
            print("test_fgt_loader_img:", images.shape, labels.shape)
            break


    def loss_f(self, merged_model, inputs, targets):
        outputs = merged_model(inputs)
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
        
        self.loader = self.train_fgt_loader_img
        self.class_to_remove = class_to_remove
        self.Truncatedmodel = TruncatedResNet(self.net).to(opt.device)
        self.Remainingmodel = RemainingResNet(self.net).to(opt.device)
        merged_model = copy.deepcopy(self.net)

        
        for images, labels in self.train_retain_loader_img:
            print("train_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.train_fgt_loader_img:
            print("train_fgt_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_retain_loader_img:
            print("test_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_fgt_loader_img:
            print("test_fgt_loader_img:", images.shape, labels.shape)
            break
        
        
    def loss_f(self, merged_model, inputs_r, targets_r):
        loss = self.criterion(merged_model(inputs_r), targets_r) * (-1)
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


        for images, labels in self.train_retain_loader_img:
            print("train_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.train_fgt_loader_img:
            print("train_fgt_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_retain_loader_img:
            print("test_retain_loader_img:", images.shape, labels.shape)
            break

        for images, labels in self.test_fgt_loader_img:
            print("test_fgt_loader_img:", images.shape, labels.shape)
            break
        
        merged_model = copy.deepcopy(self.net)

        # === Overwrite the parts that were trained in RemainingResNet ===
        merged_model.layer4[1].conv2 = self.Remainingmodel.layer4_1_conv2
        merged_model.layer4[1].bn2 = self.Remainingmodel.layer4_1_bn2
        merged_model.avgpool = self.Remainingmodel.avgpool
        merged_model.fc = self.Remainingmodel.fc        

    def loss_weighted(self, inputs_r, targets_r, inputs_f, targets_f, merged_model):
        retain_loss = self.criterion(merged_model(inputs_r), targets_r)
        forget_loss = self.criterion(merged_model(inputs_f), targets_f)
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
        zero_acc_patience = 50
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
        forget_loader_img = cycle(self.train_fgt_loader_img)
        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
        
        epoch_times = []

        for epoch in tqdm(range(self.epochs)):
            start_time = time.time()
            for inputs_r, targets_r in self.train_retain_loader_img:
                inputs_r, targets_r = inputs_r.to(opt.device), targets_r.to(opt.device)
                inputs_f, targets_f = next(forget_loader_img)
                inputs_f, targets_f = inputs_f.to(opt.device), targets_f.to(opt.device)

                total_loss = self.loss_weighted(inputs_r, targets_r, inputs_f, targets_f, merged_model)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
            end_time = time.time()
            duration = end_time - start_time
            epoch_times.append(duration)
            with torch.no_grad():
                self.net.eval()
                acc_train_ret = evaluate_embedding_accuracy(merged_model, self.train_retain_loader_img, opt.device)/100
                acc_train_fgt = evaluate_embedding_accuracy(merged_model, self.train_fgt_loader_img, opt.device)/100
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


class newmethod(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        self.loader = self.train_fgt_loader
        self.class_to_remove = class_to_remove

    def loss_f(self, inputs, targets):
        with torch.no_grad():
            logits = self.net.fc(inputs)
            top2 = torch.topk(logits, k=2, dim=1)
            pred = top2.indices[:, 1]  # second-best prediction (nearest incorrect)
        outputs = self.net.fc(inputs)
        loss = self.criterion(outputs, pred)
        return loss


class RetrainedEmbedding(BaseMethod):
    def __init__(self, net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=None):
        super().__init__(net, train_retain_loader, train_fgt_loader, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real)
        
        self.fc_layer = nn.Linear(512, opt.num_classes).to('cuda')

        self.optimizer = optim.SGD(self.fc_layer.parameters(), lr=opt.lr_unlearn, weight_decay=5e-5)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.4)

        if opt.dataset == 'TinyImageNet':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.1) #learning rate decay
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)


        self.class_to_remove = class_to_remove
        self.epochs = opt.epochs_unlearn



    def run(self):
        
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

        Aor = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True) * 100
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
        print(f"Best Epoch {best_epoch+1}: Train Acc {best_train_acc:.2f}%, Val Acc {best_acc:.2f}%")

        self.net.fc = self.fc_layer
        
        return self.net
    

class MaximeMethod(BaseMethod):
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
      
        
    def run(self):
        
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
    

        # def forget_class_for_yasi(M: torch.tensor,
        #                           B: torch.tensor,
        #                           p: int,
        #                           num_classes: int):
        #     M = torch.clone(M)
        #     B = torch.clone(B)
        #     q = choice([i for i in range(num_classes) if i != p])
        #     M[:, p] = M[:, q]
        #     B[p] = B[q]-1e-3
        #     return M, B
    
    
        def forget_class_for_yasi(M: torch.Tensor,
                                B: torch.Tensor,
                                forget_classes: list,
                                num_classes: int):
            M = M.clone()
            B = B.clone()
            for p in forget_classes:
                q = choice([i for i in range(num_classes) if i not in forget_classes and i != p])
                M[p, :] = M[q, :]
                B[p] = B[q] - 1e-3
            return M, B

    
    
        # Extract the last Linear layer from self.net.fc, whether it's Sequential or Linear
        def find_final_linear(module):
            if isinstance(module, nn.Linear):
                return module
            elif isinstance(module, nn.Sequential):
                for layer in reversed(module):
                    if isinstance(layer, nn.Linear):
                        return layer
            raise ValueError("No nn.Linear layer found in self.net.fc")

        final_linear = find_final_linear(self.net.fc)
        embedding_dim = final_linear.in_features
        num_classes = final_linear.out_features

        weight_np = final_linear.weight.detach().cpu().numpy()
        bias_np = final_linear.bias.detach().cpu().numpy()

        weight_tensor = torch.tensor(weight_np, dtype=torch.float32, device=opt.device)
        bias_tensor = torch.tensor(bias_np, dtype=torch.float32, device=opt.device)

        M, B = forget_class_for_yasi(weight_tensor, bias_tensor, self.class_to_remove, num_classes)

        # Widen model with an extra output for the shadow class
        widen_model = nn.Linear(embedding_dim, num_classes).to(opt.device)
        with torch.no_grad():
            widen_model.weight[:num_classes] = M
            widen_model.bias[:num_classes] = B


        Aor = calculate_accuracy(self.net, self.test_retain_loader, use_fc_only=True) * 100
        retain_count = count_samples(self.train_retain_loader)
        forget_count = count_samples(self.train_fgt_loader)
        total_count = retain_count + forget_count
                

        epoch = 0
        duration = 0
            
        retain_accuracy = evaluate_embedding_accuracy(widen_model, self.train_retain_loader, opt.device)
        forget_accuracy = evaluate_embedding_accuracy(widen_model, self.train_fgt_loader, opt.device)

        retainfull_val_acc = evaluate_embedding_accuracy(widen_model, self.retainfull_loader_real, opt.device)
        forgetfull_val_acc = evaluate_embedding_accuracy(widen_model, self.forgetfull_loader_real, opt.device)

        retaintest_val_acc = evaluate_embedding_accuracy(widen_model, self.test_retain_loader, opt.device)
        forgettest_val_acc = evaluate_embedding_accuracy(widen_model, self.test_fgt_loader, opt.device)
        

        AUS = calculate_AUS(forgettest_val_acc, retaintest_val_acc, Aor)  # Calculate AUS using the accuracies

        # wandb.log({"retainfull_val_acc": retainfull_val_acc, "forgetfull_val_acc": forgetfull_val_acc,
        #            "retaintest_val_acc": retaintest_val_acc, "forgettest_val_acc": forgettest_val_acc, "AUS": AUS})


        print(f"Train Retain Acc: {retain_accuracy:.2f}% | Train Forget Acc: {forget_accuracy:.2f}% | "
            f"Val Retain full Acc: {retainfull_val_acc:.2f}% | Val Forget full Acc: {forgetfull_val_acc:.2f}%  | "
            f"Val Retain Test Acc: {retaintest_val_acc:.2f}% | Val Forget Test Acc: {forgettest_val_acc:.2f}% | "
            f"AUS: {AUS:.2f}"
            )

                
        log_summary_across_classes(
            best_epoch=round(epoch,4),
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
                        
        # Prune the shadow class to return a normal classifier
        pruned_model = nn.Linear(embedding_dim, num_classes).to(opt.device)
        with torch.no_grad():
            pruned_model.weight = torch.nn.Parameter(widen_model.weight[:num_classes])
            pruned_model.bias = torch.nn.Parameter(widen_model.bias[:num_classes])

        self.net.fc = pruned_model
        self.model = self.net
        
        return self.model

