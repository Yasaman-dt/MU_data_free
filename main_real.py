from copy import deepcopy
import pandas as pd
from error_propagation import Complex
from utils import accuracy, set_seed, get_retrained_model, get_trained_model
from MIA_code.MIA import get_MIA_SVC
from opts import OPT as opt
import time
from Unlearning_methods_real import choose_method
from error_propagation import Complex
import os
import torch
import numpy as np
from generate_emb_samples_resnet18 import generate_emb_samples
from create_embeddings_utils import get_model
from torch.utils.data import TensorDataset, DataLoader
from Unlearning_methods_real import calculate_accuracy

DATASET_NUM_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "TinyImageNet": 200,
}

dataset_name_lower = opt.dataset

if dataset_name_lower.lower() in ["cifar10", "cifar100"]:
    dataset_name_upper = dataset_name_lower.upper()
else:
    dataset_name_upper = dataset_name_lower  # keep original capitalization for "tinyImagenet"

model_name = opt.model.upper()
num_classes = DATASET_NUM_CLASSES[dataset_name_upper]
n_model = opt.n_model

DIR = "/projets/Zdehghani/MU_data_free"
weights_folder = "weights"

matrix_B_224 = f"{DIR}/{weights_folder}/chks_{dataset_name_lower}/original/matrix_B_224_m{n_model}.npy"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings_folder = "embeddings"

def AUS(a_t, a_or, a_f):
    aus=(Complex(1, 0)-(a_or-a_t))/(Complex(1, 0)+abs(a_f))
    return aus

  
def main(train_retain_loader_real, train_fgt_loader_real, test_retain_loader, test_fgt_loader, train_loader=None, test_loader=None, seed=0, class_to_remove=0):
   
    v_orig, v_unlearn, v_rt = None, None, None
    original_pretr_model = get_trained_model()

    original_model = deepcopy(original_pretr_model)

    #original_pretr_model = original_pretr_model_total.fc
    original_model.to(opt.device)
    original_model.eval()
    opt.class_to_rem_curr = class_to_remove
    if opt.run_original:
        if opt.mode =="CR":
             # df_or_model = pd.DataFrame([0],columns=["PLACEHOLDER"])
             df_or_model = get_MIA_SVC(train_loader=None, test_loader=None,model=original_model,opt=opt,fgt_loader=train_fgt_loader_real,fgt_loader_t=test_fgt_loader)
             df_or_model["forget_test_accuracy"] = calculate_accuracy(original_model, test_fgt_loader, use_fc_only=True)
             df_or_model["retain_test_accuracy"] = calculate_accuracy(original_model, test_retain_loader, use_fc_only=True)

        df_or_model["forget_accuracy"] = calculate_accuracy(original_model, train_fgt_loader_real, use_fc_only=True)
        df_or_model["retain_accuracy"] = calculate_accuracy(original_model, train_retain_loader_real, use_fc_only=True)
        #print(df_or_model)
        v_orig= df_or_model.mean(0)
        #convert v_orig back to df
        v_orig = pd.DataFrame(v_orig).T
    #print(df_or_model)

    if opt.run_unlearn:
        print('\n----BEGIN UNLEARNING----')
        pretr_model = deepcopy(original_pretr_model)
        pretr_model.to(opt.device)
        pretr_model.eval()

        timestamp1 = time.time()

        # Step 1: Generate synthetic retain samples in feature space
        data_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_full_m{n_model}.npz"
    
        data = np.load(data_path)
        embeddings_real = data["embeddings"]  # Shape: (N, 512)
        labels_real = data["labels"]  # Shape: (N,)
    
        print(f"Loaded Embeddings: {embeddings_real.shape}, Labels: {labels_real.shape}")
    
        # Convert to tensors
        embeddings_tensor_real = torch.tensor(embeddings_real, dtype=torch.float32)
        labels_tensor_real = torch.tensor(labels_real, dtype=torch.long)
    
        # Split into forget and retain sets
        forget_mask_real = labels_tensor_real == forget_class
        retain_mask_real = labels_tensor_real != forget_class
    
        # Forget set (samples from class 0)
        forget_embeddings_real = embeddings_tensor_real[forget_mask_real]
        forget_labels_real = labels_tensor_real[forget_mask_real]
    
        # Retain set (samples from all other classes)
        retain_embeddings_real = embeddings_tensor_real[retain_mask_real]
        retain_labels_real = labels_tensor_real[retain_mask_real]
    
        print(f"Forget set size: {forget_embeddings_real.shape}, Retain set size: {retain_embeddings_real.shape}")
    
        # Create DataLoaders for validation
        forgetfull_loader_real = DataLoader(TensorDataset(forget_embeddings_real, forget_labels_real), batch_size, shuffle=False)
        retainfull_loader_real = DataLoader(TensorDataset(retain_embeddings_real, retain_labels_real), batch_size, shuffle=False)
            

            
        if opt.mode == "CR":
            #set tollerance for stopping criteria
            opt.target_accuracy = 0.00
            #approach = choose_method(opt.method)(pretr_model, retain_loader_synth, forget_loader_synth, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=class_to_remove)  #generated samples
            approach = choose_method(opt.method)(pretr_model, train_retain_loader_real, train_fgt_loader_real, test_retain_loader, test_fgt_loader, retainfull_loader_real, forgetfull_loader_real, class_to_remove=class_to_remove) #real samples

        if opt.load_unlearned_model:
            print("LOADING UNLEARNED MODEL")
            if opt.mode == "CR":
                unlearned_model_dict = torch.load(f"{opt.root_folder}/out_real/{opt.mode}/{opt.dataset}/{opt.method}/lr{opt.lr_unlearn}/models/unlearned_model_{opt.method}_seed_{seed}_m{n_model}_class_{'_'.join(map(str, class_to_remove))}.pth")

            unlearned_model = get_trained_model().to(opt.device)
            unlearned_model.load_state_dict(unlearned_model_dict)
            print("UNLEARNED MODEL LOADED")
        else:
            unlearned_model = approach.run()

        unlearned_model.eval()
        #save model
        #if opt.save_model:
        #    if opt.mode == "CR":
        #        torch.save(unlearned_model.state_dict(), f"{opt.root_folder}/out_real/{opt.mode}/{opt.dataset}/{opt.method}/lr{opt.lr_unlearn}/models/unlearned_model_{opt.method}_m{n_model}_seed_{seed}_class_{'_'.join(map(str, class_to_remove))}.pth")
#
        unlearn_time = time.time() - timestamp1
        print("BEGIN SVC FIT")

        if opt.mode == "CR":
            df_un_model = get_MIA_SVC(train_loader=None, test_loader=test_loader,model=unlearned_model.fc,opt=opt,fgt_loader=train_fgt_loader_real,fgt_loader_t=test_fgt_loader)
            print('F1 mean: ',df_un_model.F1.mean())
            #df_un_model = pd.DataFrame([0],columns=["PLACEHOLDER"])

    
        df_un_model["unlearn_time"] = unlearn_time

        print("UNLEARNING COMPLETED, COMPUTING ACCURACIES...")      

        if opt.mode == "CR":
            df_un_model["forget_test_accuracy"] = calculate_accuracy(unlearned_model, test_fgt_loader, use_fc_only=True)
            df_un_model["retain_test_accuracy"] = calculate_accuracy(unlearned_model, test_retain_loader, use_fc_only=True)
            print(f'forget test acc: {df_un_model["forget_test_accuracy"]}, retain test acc: {df_un_model["retain_test_accuracy"]}')

        df_un_model["forget_accuracy"] = calculate_accuracy(unlearned_model, train_fgt_loader_real, use_fc_only=True)
        df_un_model["retain_accuracy"] = calculate_accuracy(unlearned_model, train_retain_loader_real, use_fc_only=True)
        #print(df_un_model)
        v_unlearn=df_un_model.mean(0)
        v_unlearn = pd.DataFrame(v_unlearn).T
        print("UNLEARN COMPLETED")

    #if opt.run_rt_model:
    #    print('\n----MODEL RETRAINED----')
#
    #    rt_model = get_retrained_model()
    #    rt_model.to(opt.device)
    #    rt_model.eval()
    #    if opt.mode == "CR":
    #        #df_rt_model = pd.DataFrame([0],columns=["PLACEHOLDER"])
    #        df_rt_model = get_MIA_SVC(train_loader=None, test_loader=None,model=rt_model,opt=opt,fgt_loader=train_fgt_loader,fgt_loader_t=test_fgt_loader)
    #        df_rt_model["forget_test_accuracy"] = accuracy(rt_model, test_fgt_loader, use_fc_only=True)
    #        df_rt_model["retain_test_accuracy"] = accuracy(rt_model, test_retain_loader, use_fc_only=True)
#
    #    df_rt_model["forget_accuracy"] = accuracy(rt_model, train_fgt_loader, use_fc_only=True)
    #    df_rt_model["retain_accuracy"] = accuracy(rt_model, train_retain_loader, use_fc_only=True)
#
    #    v_rt = df_rt_model.mean(0)
    #    v_rt = pd.DataFrame(v_rt).T
       


    if opt.run_unlearn:
        if opt.save_df:
            if opt.mode == "CR":
                v_unlearn.to_csv(f"{opt.root_folder}/out_real/{opt.mode}/{opt.dataset}/{opt.method}/lr{opt.lr_unlearn}/dfs/{opt.method}_m{n_model}_seed_{seed}_class_{'_'.join(map(str, class_to_remove))}.csv")
    return v_orig, v_unlearn, v_rt

if __name__ == "__main__":
    df_unlearned_total=[]
    df_retrained_total=[]
    df_orig_total=[]
    
    #create output folders
    if not os.path.exists(f"{opt.root_folder}/out_real/{opt.mode}/{opt.dataset}/{opt.method}/lr{opt.lr_unlearn}/models"):
        #os.makedirs(opt.root_folder+"out_real/"+opt.mode+"/"+opt.dataset+"/models")
        os.makedirs(f"{opt.root_folder}/out_real/{opt.mode}/{opt.dataset}/{opt.method}/lr{opt.lr_unlearn}/models")
    if not os.path.exists(f"{opt.root_folder}/out_real/{opt.mode}/{opt.dataset}/{opt.method}/lr{opt.lr_unlearn}/dfs"):
        #os.makedirs(opt.root_folder+"out_real/"+opt.mode+"/"+opt.dataset+"/dfs")
        os.makedirs(f"{opt.root_folder}/out_real/{opt.mode}/{opt.dataset}/{opt.method}/lr{opt.lr_unlearn}/dfs")

    for i in opt.seed:
        set_seed(i)

        print(f"Seed {i}")
        if opt.mode == "CR":
            for class_to_remove in opt.class_to_remove:
                print(f'------------class {class_to_remove}-----------')
                batch_size = opt.batch_size
                forget_class = class_to_remove[0]
                train_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_train_m{n_model}.npz"
                
                if dataset_name_lower == "TinyImageNet":
                    test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_val_m{n_model}.npz"
                else:
                    test_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_test_m{n_model}.npz"
                full_path = f"{DIR}/{embeddings_folder}/{dataset_name_upper}/resnet18_full_m{n_model}.npz"

                train_embeddings_data = np.load(train_path)
                test_embeddings_data = np.load(test_path)
                full_embeddings_data = np.load(full_path)


                # Access the embeddings and labels
                train_emb = train_embeddings_data["embeddings"]  # The embeddings for the training data
                train_labels = train_embeddings_data["labels"]   # The labels for the training data

                test_emb = test_embeddings_data["embeddings"]  # The embeddings for the training data
                test_labels = test_embeddings_data["labels"]   # The labels for the training data

                full_emb = full_embeddings_data["embeddings"]  # The embeddings for the training data
                full_labels = full_embeddings_data["labels"]   # The labels for the training data
                
                train_emb = torch.tensor(train_emb, dtype=torch.float32)
                train_labels = torch.tensor(train_labels, dtype=torch.long)

                test_emb = torch.tensor(test_emb, dtype=torch.float32)
                test_labels = torch.tensor(test_labels, dtype=torch.long)

                full_emb = torch.tensor(full_emb, dtype=torch.float32)
                full_labels = torch.tensor(full_labels, dtype=torch.long)
                
                # Move the tensors to the device (GPU or CPU)
                train_emb = train_emb.to(device)
                train_labels = train_labels.to(device)

                test_emb = test_emb.to(device)
                test_labels = test_labels.to(device)

                full_emb = full_emb.to(device)
                full_labels = full_labels.to(device)
                
                # Create a DataLoader for training data
                train_dataset = TensorDataset(train_emb, train_labels)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Create a DataLoader for testing data
                test_dataset = TensorDataset(test_emb, test_labels)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                full_dataset = TensorDataset(full_emb, full_labels)
                full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
                
                # -------------------- Separate Forget and Retain Sets --------------------
                # Forget set
                forget_mask_train = (train_labels == forget_class)
                forget_features_train = train_emb[forget_mask_train]
                forget_labels_train = train_labels[forget_mask_train]


                # Retain set
                retain_mask_train = (train_labels != forget_class)
                retain_features_train = train_emb[retain_mask_train]
                retain_labels_train = train_labels[retain_mask_train]
                
                # Forget set
                forget_mask_test = (test_labels == forget_class)
                forget_features_test = test_emb[forget_mask_test]
                forget_labels_test = test_labels[forget_mask_test]
                
                # Retain set
                retain_mask_test = (test_labels != forget_class)
                retain_features_test = test_emb[retain_mask_test]
                retain_labels_test = test_labels[retain_mask_test]
                
                # Forget set
                forget_mask_full = (full_labels == forget_class)
                forget_features_full = full_emb[forget_mask_full]
                forget_labels_full = full_labels[forget_mask_full]
                
                # Retain set
                retain_mask_full = (full_labels != forget_class)
                retain_features_full = full_emb[retain_mask_full]
                retain_labels_full = full_labels[retain_mask_full]
                
                # -------------------- Create DataLoaders for Forget and Retain Sets --------------------
                # Create TensorDatasets for each subset
                train_forget_dataset = TensorDataset(forget_features_train, forget_labels_train)
                train_retain_dataset = TensorDataset(retain_features_train, retain_labels_train)
                
                test_forget_dataset = TensorDataset(forget_features_test, forget_labels_test)
                test_retain_dataset = TensorDataset(retain_features_test, retain_labels_test)
                
                full_forget_dataset = TensorDataset(forget_features_full, forget_labels_full)
                full_retain_dataset = TensorDataset(retain_features_full, retain_labels_full)
                
                # Create DataLoader for each subset
                train_fgt_loader_real = DataLoader(train_forget_dataset, batch_size=batch_size, shuffle=True)
                train_retain_loader_real = DataLoader(train_retain_dataset, batch_size=batch_size, shuffle=True)
                
                test_fgt_loader_real = DataLoader(test_forget_dataset, batch_size=batch_size, shuffle=False)
                test_retain_loader_real = DataLoader(test_retain_dataset, batch_size=batch_size, shuffle=False)

                full_forget_loader = DataLoader(full_forget_dataset, batch_size=batch_size, shuffle=False)
                full_retain_loader = DataLoader(full_retain_dataset, batch_size=batch_size, shuffle=False)
                
                
                all_train_loader = train_loader
                all_test_loader = test_loader

                
                            
                opt.RT_model_weights_path = opt.root_folder+f'weights/chks_{dataset_name_lower}/retrained/best_checkpoint_without_{class_to_remove[0]}.pth'
                print(opt.RT_model_weights_path)
                
                
                row_orig, row_unl, row_ret=main(train_retain_loader_real=train_retain_loader_real,
                                                train_fgt_loader_real=train_fgt_loader_real,
                                                test_retain_loader=test_retain_loader_real,
                                                test_fgt_loader=test_fgt_loader_real,
                                                train_loader=all_train_loader,
                                                test_loader=all_test_loader,
                                                seed=i,
                                                class_to_remove=class_to_remove)

                #print results
                

                
                if row_orig is not None:
                    print(f"Original retain test acc: {row_orig['retain_test_accuracy']}")
                    df_orig_total.append(row_orig)
                if row_unl is not None:
                    print(f"Unlearned retain test acc: {row_unl['retain_test_accuracy']}")
                    df_unlearned_total.append(row_unl)
                if row_ret is not None:
                    print(f"Retrained retain test acc: {row_ret['retain_test_accuracy']}")
                    df_retrained_total.append(row_ret)
        
    print(opt.dataset)
    #create results folder if doesn't exist
    
    dfs = {"orig":[], "unlearned":[], "retrained":[]}
    for name, df in zip(dfs.keys(),[df_orig_total, df_unlearned_total, df_retrained_total]):
        if df:
            print("{name} \n")
            #merge list of pd dataframes
            dfs[name] = pd.concat(df)

            means = dfs[name].mean()
            std_devs = dfs[name].std()
            output = "\n".join([f"{col}: {100*mean:.2f} \\pm {100*std:.2f}" if col != 'unlearning_time' else f"{col}: {mean:.2f} \\pm {std:.2f}" for col, mean, std in zip(means.index, means, std_devs)])
            print(output)


            if opt.mode == "CR":
                a_t = Complex(means["retain_test_accuracy"], std_devs["retain_test_accuracy"])
                a_f = Complex(means["forget_test_accuracy"], std_devs["forget_test_accuracy"])
                a_or = opt.a_or[opt.dataset][1]
            aus = AUS(a_t, a_or, a_f)
            dfs[name]["AUS"] = aus.value
            print(f"AUS: {aus.value:.4f} \pm {aus.error:.4f}")
   
