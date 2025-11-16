import torch 
import os
import argparse
from error_propagation import Complex

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--mode", type=str, default="CR")
    parser.add_argument("--cuda", type=int, default=0, help="Select zero-indexed cuda device. -1 to use CPU.")
    
    parser.add_argument("--load_unlearned_model",action='store_true')
    
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--save_df", action='store_true')
   
    parser.add_argument("--run_original", action='store_true')
    parser.add_argument("--run_unlearn", action='store_true')
    parser.add_argument("--run_rt_model", action='store_true')

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--method", type=str, default="SCAR")

    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--bsize", type=int, default=1024)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=30, help='Num of epochs, for unlearning algorithms it is the max num of epochs') # <------- epochs train
    parser.add_argument("--scheduler", type=int, nargs='+', default=[25,40])
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--lambda_1", type=float, default=1)
    parser.add_argument("--lambda_2", type=float, default=5)

    parser.add_argument("--delta", type=float, default=.5)
    parser.add_argument("--gamma1", type=float, default=3)
    parser.add_argument("--gamma2", type=float, default=3)

    parser.add_argument("--patience", type=int, default=10, help="Number of epochs with no improvement to wait before stopping.")
    parser.add_argument("--samples_per_class", type=int, default=1000, help="Number of synthetic samples per class to generate for unlearning.")
    parser.add_argument("--n_model", type=int, default=0, help="Model number")
    
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'bernoulli', 'uniform', 'laplace', 'gumbel'],
                        help='Type of noise distribution for synthetic feature generation')


    
    parser.add_argument(
        "--forget_mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Whether to run single-class or multi-class unlearning."
    )

    
    parser.add_argument(
        "--num_forget_classes",
        type=int,
        default=2,
        help="Number of classes to forget in multi-class unlearning."
    )    
    
    options = parser.parse_args()
    return options


class OPT:
    args = get_args()
    print(args)
    run_name = args.run_name
    dataset = args.dataset
    patience = args.patience
    samples_per_class = args.samples_per_class
    n_model = args.n_model
    noise_type = args.noise_type
    forget_mode = args.forget_mode
    num_forget_classes = args.num_forget_classes

    mode = args.mode
    if args.mode == 'HR':
        seed = [0,1,2,3,4,5,6,7,8,42]
        class_to_remove = None
    else:
        seed = [42]
        if dataset == 'cifar10':
            single_class_runs = [[i*1] for i in range(10)]#[ [0], [1], ..., [9] ]#
            permutation_map = [4, 3, 5, 9, 8, 1, 0, 7, 6, 2]            
            
            if forget_mode == "single":
                class_to_remove = single_class_runs
            else: 
                forget_classes = permutation_map[:num_forget_classes] # e.g. [4, 3] if k=2
                class_to_remove = [forget_classes]    
                         
        elif dataset == 'cifar100':
            single_class_runs = [[i] for i in range(100)]
            permutation_map = [25, 58, 38, 23, 96, 54, 51, 49, 98, 66,
                            16, 52, 40, 71, 63, 79, 53, 12, 46, 55,
                            83, 27, 41, 20, 30, 14, 70, 45, 61, 29,
                            4, 39, 21, 87, 60, 68, 75, 2, 92, 5,
                            57, 42, 0, 8, 97, 31, 50, 47, 13, 80,
                            34, 91, 17, 69, 85, 76, 94, 73, 99, 74,
                            43, 67, 62, 89, 36, 65, 26, 78, 19, 11,
                            90, 15, 3, 24, 72, 18, 33, 22, 7, 88,
                            44, 56, 86, 81, 82, 1, 48, 28, 6, 64,
                            9, 32, 35, 77, 95, 84, 59, 93, 10, 37]

            if forget_mode == "single":
                class_to_remove = single_class_runs
            else: 
                forget_classes = permutation_map[:num_forget_classes]  
                class_to_remove = [forget_classes]                   
                
        elif dataset == 'TinyImageNet':
            single_class_runs = [[i*1] for i in range(200)]
            permutation_map = [126, 162, 16, 87, 88, 17, 78, 93, 127, 14,
                            26, 197, 62, 196, 151, 111, 4, 57, 117, 110,
                            61, 66, 191, 138, 101, 159, 81, 0, 46, 153,
                            192, 146, 182, 25, 105, 19, 137, 150, 98, 32,
                            91, 60, 158, 187, 140, 23, 94, 9, 129, 188,
                            39, 155, 121, 84, 52, 130, 165, 154, 31, 82,
                            47, 15, 178, 148, 118, 74, 194, 80, 27, 45,
                            190, 69, 73, 163, 59, 152, 48, 185, 67, 106,
                            169, 89, 172, 2, 180, 35, 177, 30, 7, 76,
                            139, 175, 83, 29, 109, 95, 21, 186, 119,
                            51, 70, 49, 141, 147, 125, 96, 179, 1, 142,
                            176, 92, 33, 108, 132, 6, 5, 12, 13, 79,
                            167, 145, 24, 36, 183, 164, 38, 56, 195, 122,
                            133, 107, 40, 189, 65, 68, 136, 54, 90, 115,
                            104, 11, 18, 120, 135, 85, 75, 144, 58, 55,
                            37, 113, 156, 103, 123, 199, 184, 34, 143, 173,
                            128, 166, 124, 170, 86, 63, 3, 114, 193, 10,
                            42, 102, 43, 53, 72, 116, 8, 97, 100, 112,
                            157, 41, 168, 50, 134, 77, 44, 71, 64, 161,
                            20, 160, 131, 99, 22, 198, 171, 28, 174, 181, 149]


            if forget_mode == "single":
                class_to_remove = single_class_runs
            else:
                forget_classes = permutation_map[:(num_forget_classes)]  
                class_to_remove = [forget_classes]   
                
        #class_to_remove = [[i for i in range(100)][:j] for j in [1]+[z*10 for z in range(1,10)]+[98]]
        #print('Class to remove iter. : ', class_to_remove)

    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    
    
    save_model = args.save_model
    save_df = args.save_df
    load_unlearned_model = args.load_unlearned_model



    # gets current folder path
    root_folder = os.path.dirname(os.path.abspath(__file__)) + "/"

    # Model
    model = args.model
    ### RUN model type
    run_original = args.run_original
    run_unlearn = args.run_unlearn
    run_rt_model = args.run_rt_model
    
    # Data
    data_path = os.path.expanduser('/projets/Zdehghani/MU_data_free/datasets')

    # num_retain_samp sets the percentage of retain or retain surrogate data to use during unlearning
    # the num of Samples used is bsize*num_retain_samp
    if dataset == 'cifar10':
        num_classes = 10
        num_retain_samp = 5#1 for cr
    elif dataset == 'cifar100':
        num_classes = 100
        num_retain_samp = 5#3 for cr
    elif dataset == 'TinyImageNet':
        num_classes = 200
        num_retain_samp = 90
    
    

    num_workers = args.num_workers

    method = args.method#' #NegativeGradient, RandomLabels,...
    
    # unlearning params
        
    batch_size = args.bsize
    epochs_unlearn = args.epochs
    lr_unlearn = args.lr
    wd_unlearn = args.wd
    momentum_unlearn = args.momentum
    temperature = args.temperature
    scheduler = args.scheduler

    #DUCK specific
    lambda_1 = args.lambda_1
    lambda_2 = args.lambda_2
    delta = args.delta
    gamma1 = args.gamma1
    gamma2 = args.gamma2
    target_accuracy = 0.01 
    
    #MIA specific
    iter_MIA = 5 #numo f iterations
    verboseMIA = False

    weight_file_id = '1tTdpVS3was0RTZszQfLt2tGdixwd3Oy6'
    if model== 'resnet18':
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+f'weights/chks_cifar100/original/best_checkpoint_resnet18_m{n_model}.pth'
            if mode == "CR":
                RT_model_weights_path = root_folder+f'weights/chks_cifar100/retrained/best_checkpoint_without_{class_to_remove}.pth'
        
        elif dataset== 'cifar10':
            or_model_weights_path = root_folder+f'weights/chks_cifar10/original/best_checkpoint_resnet18_m{n_model}.pth'
            if mode == "CR":
                RT_model_weights_path = root_folder+f'weights/chks_cifar10/retrained/best_checkpoint_without_{class_to_remove}.pth'

        elif dataset== 'TinyImageNet':
            or_model_weights_path = root_folder+f'weights/chks_TinyImageNet/original/best_checkpoint_resnet18_m{n_model}.pth'
            if mode == "CR":
                RT_model_weights_path = root_folder+f'weights/chks_TinyImageNet/retrained/best_checkpoint_without_{class_to_remove}.pth'
     
    elif model == 'resnet50':
        if dataset== 'cifar10':
            or_model_weights_path = root_folder+f'weights/chks_cifar10/original/best_checkpoint_resnet50_m{n_model}.pth'
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+f'weights/chks_cifar100/original/best_checkpoint_resnet50_m{n_model}.pth'
        if dataset== 'TinyImageNet':
            or_model_weights_path = root_folder+f'weights/chks_TinyImageNet/original/best_checkpoint_resnet50_m{n_model}.pth'
    
    elif model == 'resnet34':
        if dataset== 'cifar10':
            or_model_weights_path = root_folder+f'weights/chks_cifar10/original/best_checkpoint_resnet34_m{n_model}.pth'
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+f'weights/chks_cifar100/original/best_checkpoint_resnet34_m{n_model}.pth'
        if dataset== 'TinyImageNet':
            or_model_weights_path = root_folder+f'weights/chks_TinyImageNet/original/best_checkpoint_resnet34_m{n_model}.pth'
    
    elif model == 'ViT':
        if dataset== 'cifar10':
            or_model_weights_path = root_folder+f'weights/chks_cifar10/original/best_checkpoint_ViT_m{n_model}.pth'
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+f'weights/chks_cifar100/original/best_checkpoint_ViT_m{n_model}.pth'
        if dataset== 'TinyImageNet':
            or_model_weights_path = root_folder+f'weights/chks_TinyImageNet/original/best_checkpoint_ViT_m{n_model}.pth'

    elif model == 'swint':
        if dataset == 'cifar10':
            or_model_weights_path = root_folder+f'weights/chks_cifar10/original/best_checkpoint_swint_m{n_model}.pth'
        elif dataset == 'cifar100':
            or_model_weights_path = root_folder+f'weights/chks_cifar100/original/best_checkpoint_swint_m{n_model}.pth'
        elif dataset == 'TinyImageNet':
            or_model_weights_path = root_folder+f'weights/chks_TinyImageNet/original/best_checkpoint_swint_m{n_model}.pth'

    elif model == 'AllCNN':
        if dataset== 'cifar10':
            or_model_weights_path = root_folder+f'weights/chks_cifar10/original/best_checkpoint_AllCNN_m{n_model}.pth'
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+f'weights/chks_cifar100/original/best_checkpoint_AllCNN_m{n_model}.pth'
        if dataset== 'TinyImageNet':
            or_model_weights_path = root_folder+f'weights/chks_TinyImageNet/original/best_checkpoint_tiny_AllCNN_m{n_model}.pth'
    else:
        raise NotImplementedError
    

