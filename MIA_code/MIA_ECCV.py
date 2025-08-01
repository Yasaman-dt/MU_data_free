import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
import os

iter_MIA = 20

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd

def compute_f1_score(predicted_labels, true_labels):
    """
    Compute the F1 score given the predicted labels and true labels.

    Args:
        predicted_labels (list or array-like): The predicted labels.
        true_labels (list or array-like): The true labels.

    Returns:
        float: The F1 score.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for predicted, true in zip(predicted_labels, true_labels):
        if predicted == 1 and true == 1:
            true_positives += 1
        elif predicted == 1 and true == 0:
            false_positives += 1
        elif predicted == 0 and true == 1:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision,recall,f1_score

#function to compute accuracy given a model and a dataloader
def compute_accuracy(model, dataloader,device,targ_val = None):
    model.eval()
    correct = 0
    total = 0 
    prediction_F1 = []
    labels = []
    for inputs, targets in dataloader:
        if targ_val is None:
            inputs, targets = inputs.to(device), targets.to(device)
        else:
            inputs, targets = inputs[targets==targ_val].to(device), targets[targets==targ_val].to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        prediction_F1.extend(predicted.detach().cpu().numpy())
        labels.extend(targets.detach().cpu().numpy())
    #compute F1 score
    if targ_val is None:
        labels= np.asarray(labels)
        chance = labels.sum()/labels.shape[0]
        P,R,F1 = precision_recall_fscore_support(np.asarray(labels).astype(np.int64), np.asarray(prediction_F1).astype(np.int64), average='binary')[:3]
        return correct / total, chance ,P,R,F1
    else:
        return correct / total
    
    
def compute_mutual_information(vector1, vector2):
    import math
    """
    Compute the mutual information between two binary vectors.

    Args:
        vector1 (list or array-like): The first binary vector.
        vector2 (list or array-like): The second binary vector.

    Returns:
        float: The mutual information.
    """
    assert len(vector1) == len(vector2), "Vectors must have the same length."
    
    n = len(vector1)
    
    # Calculate the joint probability distribution
    joint_prob = [[0, 0], [0, 0]]
    
    for i in range(n):
        joint_prob[vector1[i]][vector2[i]] += 1
    
    joint_prob = [[count / n for count in row] for row in joint_prob]
    print(joint_prob)
    # Calculate the marginal probability distributions
    marg_prob1 = [sum(joint_prob[i][j] for j in range(2)) for i in range(2)]
    marg_prob2 = [sum(joint_prob[i][j] for i in range(2)) for j in range(2)]
    
    # Calculate the mutual information
    mutual_info = 0
    
    for i in range(2):
        for j in range(2):
            if joint_prob[i][j] > 0 and marg_prob1[i] > 0 and marg_prob2[j] > 0:
                mutual_info += joint_prob[i][j] * \
                    (math.log(joint_prob[i][j] /
                              (marg_prob1[i] * marg_prob2[j]), 2))
    
    return mutual_info

def compute_accuracy_SVC(predicted, labels, targ_val = None):
    if targ_val is not None:
        predicted, labels = predicted[labels==targ_val], labels[labels==targ_val]

    total = len(predicted)
    correct = np.equal(predicted,labels).sum().item()
    #compute F1 score
    if targ_val is None:
        labels= np.asarray(labels)
        chance = labels.sum()/labels.shape[0]
        P,R,F1 = precision_recall_fscore_support(np.asarray(labels).astype(np.int64), np.asarray(predicted).astype(np.int64), average='micro')[:3]
        mutual = compute_mutual_information(np.asarray(labels).astype(np.int64), np.asarray(predicted).astype(np.int64))
        return correct / total, chance ,P,R,F1, mutual
    else:
        return correct / total

def training_SVC(model,X_train, X_test, z_train, z_test, device):
    param_grid = {'C': [1,5,10,100],
              'gamma': [1, 0.1, 0.01], 
              'kernel': ['rbf']}
    grid = GridSearchCV(model, param_grid, refit = True, verbose=0, cv=3, n_jobs=4) 
    grid.fit(X_train, z_train)
    best_model = grid.best_estimator_
    print(grid.best_params_,grid.best_score_)

    results = best_model.predict(X_test)

    accuracy,chance, precision, recall,F1, mutual = compute_accuracy_SVC(results, z_test)
    accuracy_test_ex = compute_accuracy_SVC(results, z_test,0)
    accuracy_train_ex = compute_accuracy_SVC(results, z_test,1)

    print(f'Test F1: {round(F1,3)}')
    return accuracy,chance,accuracy_test_ex,accuracy_train_ex, precision, recall,F1, mutual


def collect_prob(data_loader, model, device):
    
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=256, shuffle=True)
    prob = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            batch = [tensor.to(device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=1).data)
        prob=torch.cat(prob)            
    return prob


def collect_prob2(data_loader, model, forget_class, device):
    
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=256, shuffle=True)
    prob = []


    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            batch = [tensor.to(device) for tensor in batch]
            data, target = batch
            output = model(data)[target==forget_class]
            if output.shape[0]>0:
                prob.append(F.softmax(output, dim=1).data)
        prob=torch.cat(prob)            
    return prob

def get_membership_attack_data(train_loader, test_loader, model):    
    #get membership attack data, this function will return X_r, Y_r, X_f, Y_f


    train_prob = collect_prob(train_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_tr = train_prob.cpu()
    Y_tr = torch.zeros(len(train_prob),dtype=torch.int64)
    
    X_te = test_prob.cpu()
    Y_te = torch.ones(len(test_prob),dtype=torch.int64)


    N_tr = X_tr.shape[0]
    N_te = X_te.shape[0]

    #sample from training data N_r samples
    Idx = np.arange(N_tr)
    np.random.shuffle(Idx)
    X_tr = X_tr[Idx[:N_te],:]
    Y_tr = Y_tr[Idx[:N_te]]
    N_tr = X_tr.shape[0]

    xtrain = torch.cat([X_tr[:int(0.8*N_tr)],X_te[:int(0.8*N_te)]],dim=0)
    ytrain = torch.cat([Y_tr[:int(0.8*N_tr)],Y_te[:int(0.8*N_te)]],dim=0)

    xtest = torch.cat([X_tr[int(0.8*N_tr):],X_te[int(0.8*N_te):]],dim=0) 
    ytest = torch.cat([Y_tr[int(0.8*N_tr):],Y_te[int(0.8*N_te):]],dim=0)

    # Compute entropy
    entropy = torch.sum(-train_prob*torch.log(torch.clamp(train_prob,min=1e-5)),dim=1)
    train_entropy = torch.mean(entropy).item()
    train_entropy_std = torch.std(entropy).item()
    print(f"train entropy: {train_entropy} +- {train_entropy_std}")
    entropy=torch.sum(-test_prob*torch.log(torch.clamp(test_prob,min=1e-5)),dim=1)
    test_entropy = torch.mean(entropy).item()
    test_entropy_std = torch.std(entropy).item()
    print(f"test entropy: {test_entropy} +- {test_entropy_std}")


    return xtrain,ytrain,xtest,ytest,train_entropy, test_entropy





def get_membership_attack_data_CR(fgt_loader,fgt_loader_t,model, dataset, device=None, test_data=None):   
    
        
    mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'TinyImageNet': (0.485, 0.456, 0.406),
            }

    std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'TinyImageNet': (0.229, 0.224, 0.225),
            }

    list_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset],std[dataset]),
        ]
    transform_test= transforms.Compose(list_test)



    transform_dset = transforms.Compose(
            [   transforms.RandomCrop(64, padding=8) if dataset == 'TinyImageNet' else transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean[dataset],std[dataset]),
            ]
        )
        
              
     
    #get membership attack data, this function will return X_r, Y_r, X_f, Y_f

    fgt_loader.dataset.transform = transform_test
    #print(fgt_loader.dataset.transform)

    fgt_prob = collect_prob(fgt_loader, model, device)
    fgt_prob_t = collect_prob(fgt_loader_t, model, device)

    X_fgt = fgt_prob.cpu()
    Y_fgt = torch.zeros(len(fgt_prob),dtype=torch.int64)

    X_fgt_t = fgt_prob_t.cpu()
    Y_fgt_t = torch.ones(len(fgt_prob_t),dtype=torch.int64)


    
    N_fgt_t = X_fgt_t.shape[0]


    np.random.shuffle(X_fgt_t)
    np.random.shuffle(X_fgt)
    X_fgt = X_fgt[:N_fgt_t*3,:]
    Y_fgt = Y_fgt[:N_fgt_t*3]
    N_fgt = X_fgt.shape[0]


    n_samp = int(0.8*N_fgt) 
    n_sampt = int(0.8*N_fgt_t)

    xtrain = torch.cat([X_fgt_t[:n_sampt,:],X_fgt[:n_samp,:]],dim=0)
    ytrain = torch.cat([Y_fgt_t[:n_sampt],Y_fgt[:n_samp]],dim=0)

    xtest = torch.cat([X_fgt_t[n_sampt:,:],X_fgt[n_samp:,:]],dim=0)
    ytest = torch.cat([Y_fgt_t[n_sampt:],Y_fgt[n_samp:]],dim=0)


    # Compute entropy
    entropy=torch.sum(-fgt_prob*torch.log(torch.clamp(fgt_prob,min=1e-5)),dim=1)
    fgt_entropy = torch.mean(entropy).item()
    fgt_entropy_std = torch.std(entropy).item()
    print(f"fgt entropy: {fgt_entropy} +- {fgt_entropy_std}")
    entropy=torch.sum(-fgt_prob_t*torch.log(torch.clamp(fgt_prob_t,min=1e-5)),dim=1)
    fgt_entropy_t = torch.mean(entropy).item()
    fgt_entropy_std = torch.std(entropy).item()
    print(f"fgt_t entropy: {fgt_entropy_t} +- {fgt_entropy_std}")


    print('check input vectors: ',xtrain.shape,xtest.shape,np.unique(ytrain,return_counts=True))
    return xtrain,ytrain,xtest,ytest

def get_MIA_SVC(train_loader, test_loader, model, fgt_loader=None, fgt_loader_t=None, device=None, dataset=None):
    results = []
    for i in range(iter_MIA):
        train_data, train_labels, test_data, test_labels = get_membership_attack_data_CR(
            fgt_loader=fgt_loader,
            fgt_loader_t=fgt_loader_t,
            model=model,
            dataset=dataset,
            device=device,
            test_data=test_loader
        )
        model_SVC = SVC( tol = 1e-4, max_iter=4000, random_state=i) #class_weight='balanced'
        accuracy, chance,accuracy_test_ex,accuracy_train_ex, P,R,F1, mutual = training_SVC(model_SVC, train_data, test_data, train_labels, test_labels, device)
        results.append(np.asarray([accuracy, chance,accuracy_test_ex,accuracy_train_ex,P,R,F1, mutual,0, 0]))
    
    results = np.asarray(results)
    df = pd.DataFrame(results,columns=['accuracy','chance','acc | test ex','acc | train ex','precision','recall','F1', 'Mutual', "Train Entropy", "Test Entropy"])
    return df

