import logging
import sys
import torch
from logging import basicConfig, getLogger
import os
from create_embeddings_utils import (
    DATASETS,
    MODELS,
    CustomBackboneModel,
    CustomDatasetLoader,
    save_embeddings_to_npz,
    vit_input_transforms,
)
from tqdm.auto import tqdm

#DIR = "C:/Users/AT56170/Desktop/Codes/Machine Unlearning - Classification/MU_data_free"
DIR = "/projets/Zdehghani/MU_data_free"

folder = "embeddings"
datasets_folder = "datasets"
n_model = "1"



def main(download: bool = True, *_, **__):  # Set download to True
    s_handler = logging.StreamHandler(stream=sys.stdout)
    f_handler = logging.FileHandler(filename="create_embeddings.log")
    handlers = [s_handler, f_handler]
    basicConfig(level=logging.INFO, handlers=handlers)
    logger = getLogger(__name__)

    for dataset_name in DATASETS.keys():
        dataset_path = f"{DIR}/{datasets_folder}/{dataset_name}"
        logger.info(f"Dataset path: {dataset_path}")  # Log the dataset path
        logger.info(f"charging dataset {dataset_name}.")
        data = CustomDatasetLoader(dataset_name, root=f"{DIR}/{datasets_folder}/{dataset_name}", download=download)
        logger.info(f"dataset {dataset_name} loaded.")

        for model_name in tqdm(list(MODELS.keys())):  
            logger.info(f"embedding {dataset_name} through {model_name}")
            if dataset_name in ["CIFAR10", "CIFAR100"]:
                dataset_name_lower = dataset_name.lower()
            else:
                dataset_name_lower = dataset_name  # keep original capitalization for "tinyImagenet"

            if model_name == 'ViT':
                checkpoint_path = f"{DIR}/weights/chks_{dataset_name_lower}/original/best_checkpoint_ViT_m{n_model}.pth"
                # Override transforms to 224 for ViT to match training:contentReference[oaicite:9]{index=9}
                t_train, t_test = vit_input_transforms(dataset_name)
                data.train_dataset.transform = t_train
                data.test_dataset.transform = t_test
                if hasattr(data, 'val_dataset'):
                    data.val_dataset.transform = t_test
                # Also update the concat’d dataset’s internal datasets, if present
                if hasattr(data, 'dataset') and hasattr(data.dataset, 'datasets'):
                    for d in data.dataset.datasets:
                        d.transform = t_test
            else:
                checkpoint_path = f"{DIR}/weights/chks_{dataset_name_lower}/original/best_checkpoint_{model_name}_m{n_model}.pth"

            model = CustomBackboneModel(model_name, dataset_name, checkpoint_path=checkpoint_path)      
            if not os.path.exists(checkpoint_path):
                logger.warning(f"Checkpoint not found at {checkpoint_path}. Skipping...")
                continue  # Skip to next model
            # Get embeddings for train, test, and full datasets
            if dataset_name == "TinyImageNet":
                train_embeddings, train_labels, test_embeddings, test_labels, val_embeddings, val_labels, full_embeddings, full_labels = model.embed_dataset(data)
                
                # Save the train embeddings
                save_embeddings_to_npz(train_embeddings, 
                                       train_labels, 
                                       DIR, 
                                       folder, 
                                       dataset_name, 
                                       n_model,
                                       f"{model_name}_train")
                
                # Save the test embeddings
                save_embeddings_to_npz(test_embeddings, 
                                       test_labels, 
                                       DIR, 
                                       folder, 
                                       dataset_name, 
                                       n_model,
                                       f"{model_name}_test")
                
                # Save the validation embeddings
                save_embeddings_to_npz(val_embeddings, 
                                       val_labels, 
                                       DIR, 
                                       folder, 
                                       dataset_name, 
                                       n_model,
                                       f"{model_name}_val")
                
                # Save the full embeddings (train + test + val)
                save_embeddings_to_npz(full_embeddings, 
                                       full_labels, 
                                       DIR, 
                                       folder, 
                                       dataset_name, 
                                       n_model,
                                       f"{model_name}_full")
                
            elif dataset_name in ["CIFAR10", "CIFAR100"]:
                train_embeddings, train_labels, test_embeddings, test_labels, full_embeddings, full_labels = model.embed_dataset(data)
                
                # Save the train embeddings
                save_embeddings_to_npz(train_embeddings, 
                                       train_labels, 
                                       DIR, 
                                       folder, 
                                       dataset_name,
                                       n_model,
                                       f"{model_name}_train")
                
                # Save the test embeddings
                save_embeddings_to_npz(test_embeddings, 
                                       test_labels, 
                                       DIR, 
                                       folder, 
                                       dataset_name, 
                                       n_model,
                                       f"{model_name}_test")
                
                # Save the full embeddings (train + test)
                save_embeddings_to_npz(full_embeddings, 
                                       full_labels, 
                                       DIR, 
                                       folder, 
                                       dataset_name, 
                                       n_model,
                                       f"{model_name}_full")
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            del model
            logger.info(f"embedding of {dataset_name} through {model_name} done.")


if __name__ == "__main__":

    kw = {
        'download': True
    }

    main(**kw)
    
    print(1)
