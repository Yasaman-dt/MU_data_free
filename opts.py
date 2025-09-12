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
    
    mode = args.mode
    if args.mode == 'HR':
        seed = [0,1,2,3,4,5,6,7,8,42]
        class_to_remove = None
    else:
        seed = [42]
        if dataset == 'cifar10':
            class_to_remove = [[i*1] for i in range(10)]#[[0]]#
        elif dataset == 'cifar100':
            class_to_remove = [[i*1] for i in range(100)]#[[0]]#
        elif dataset == 'TinyImageNet':
            class_to_remove = [[i*1] for i in range(200)] #[[0]]#
   
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
            or_model_weights_path = root_folder+f'weights/chks_cifar10/original/best_checkpoint_swin_t_m{n_model}.pth'
        elif dataset == 'cifar100':
            or_model_weights_path = root_folder+f'weights/chks_cifar100/original/best_checkpoint_swin_t_m{n_model}.pth'
        elif dataset == 'TinyImageNet':
            or_model_weights_path = root_folder+f'weights/chks_TinyImageNet/original/best_checkpoint_swin_t_m{n_model}.pth'

    elif model == 'AllCNN':
        if dataset== 'cifar10':
            or_model_weights_path = root_folder+f'weights/chks_cifar10/original/best_checkpoint_AllCNN_m{n_model}.pth'
        if dataset== 'cifar100':
            or_model_weights_path = root_folder+f'weights/chks_cifar100/original/best_checkpoint_AllCNN_m{n_model}.pth'
        if dataset== 'TinyImageNet':
            or_model_weights_path = root_folder+f'weights/chks_TinyImageNet/original/best_checkpoint_tiny_AllCNN_m{n_model}.pth'
    else:
        raise NotImplementedError
    

