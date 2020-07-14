import numpy as np
import argparse
import wandb
import datasets as ds
import torch
import networks
import common

from torch import nn
from optimizer.radam import RAdam
from quantization_common.quantization_config import get_q_config_same
from quantization_common.quantization_enums import QUANTIZATION
from quantization_common.quantization_information import get_thresholds_list
from quantization_common.quantization_instrumentation import model_coefficient_split, update_quantization_coefficient

from training.final_stage import final_stage_training
from training.search_stage import single_iteration_training_joint

from apex import amp

CR_START = 4
PROJECT_NAME = 'HMQ'


def get_arguments():
    parser = argparse.ArgumentParser(description='HMQ Retraining for ImageNet Classification')

    # General
    parser.add_argument('--log_dir', type=str, default='/data/projects/swat/users/haih/logs/',
                        help='Weights and Bias logging folder path')
    parser.add_argument('--data_dir', type=str, default='/local_datasets/image_net/', help='Dataset folder')
    parser.add_argument('--tag', type=str, default='', help='Tagging string')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataset reading')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='Dataset name', choices=['CIFAR10', 'ImageNet'])
    parser.add_argument('--network_name', type=str, default='resnet18_cifar', help='network name',
                        choices=[ 'mobilenet_v1','mobilenet_v2','resnet50'])
    parser.add_argument('--local_rank', type=int, default=0, help='Local GPU Index (for multiple GPU training)')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epoch in the search stage')
    parser.add_argument('--n_epochs_final', type=int, default=20, help='Number of epoch in the final stage')
    # Optimizer
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum value')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Optimizer weight decay value')
    parser.add_argument('--lr_start', type=float, default=0.00001,
                        help='Optimizer learning rate for model parameters only')
    parser.add_argument('--lr_activation', type=float, default=0.01,
                        help='Optimizer learning rate for activation HMQ parameters only')
    parser.add_argument('--lr_coefficient', type=float, default=0.01,
                        help='Optimizer learning rate for coefficients HMQ parameters only')
    parser.add_argument('--fp16', action='store_true',
                        help='Use Float point 16 bits for training')
    # HMQ Loss
    parser.add_argument('--gamma', type=float, default=32.0,
                        help='A Scaling factor of the Compression Loss')
    parser.add_argument('--gamma_rate', type=float, default=2.0,
                        help='A Pow factor of the Compression Loss')
    parser.add_argument('--target_compression', type=float, default=17.0,
                        help='Coefficients target compression rate')
    parser.add_argument('--target_compression_activation', type=float, default=8.0,
                        help='Activation target compression rate')
    parser.add_argument('--cycle_size', type=int, default=5,
                        help='Training Cycle size until changing target compression')
    parser.add_argument('--n_target_steps', type=int, default=4,
                        help='Number of Cycle until reaching the target compression')

    # Gumbel Softmax
    parser.add_argument('--gumbel_ste', action='store_true',
                        help='Use STE on the gumbel softmax')
    parser.add_argument('--n_gumbel', type=int, default=25,
                        help='Number until rounding of single temperature step')
    parser.add_argument('--temp_step', type=float, default=1e-2,
                        help='Exp factor of the temperature decay')

    # Quantization config
    parser.add_argument('--quantization_part', type=str, default='BOTH',
                        help='Selection which part of the network to quantize',
                        choices=['BOTH', 'ACTIVATION', 'COEFFICIENT'])
    parser.add_argument('--bits_list', type=int, nargs='+', default=[8, 7, 6, 5, 4, 3, 2, 1],
                        help='A List of bits that support by hardware for coefficients')

    parser.add_argument('--activation_max_bit_width', type=int, default=8,
                        help='Activation maximal bit width')
    parser.add_argument('--activation_min_bit_width', type=int, default=4,
                        help='Activation minimal bit width')
    parser.add_argument('--n_thresholds_shifts', type=int, default=8,
                        help='Number of threshold shift')

    args = parser.parse_args()  # parse arguments
    config = {}
    config.update({k: v for (k, v) in args._get_kwargs()})  # update via user input
    return config


def base_runner():
    #######################################
    # Search Working Device
    #######################################
    print("-" * 100)
    cc = get_arguments()
    if cc.get('local_rank') == 0:
        wandb.init(project=PROJECT_NAME, dir=cc.get('log_dir'))
        wandb.config.update(cc)  # adds all of the arguments as config variables
        print(f"W & B Log Dir:{wandb.wandb_dir()}")
    print("-" * 100)
    #######################################
    # Setting Dataset
    #######################################
    if cc.get('fp16'):
        torch.cuda.set_device(cc.get('local_rank'))
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    train_loader, test_loader, train_sampler = ds.get_dataset(ds.Dataset[cc.get('dataset')], cc.get('data_dir'),
                                                              batch_size=cc.get('batch_size'),
                                                              num_workers=cc.get('num_workers'),
                                                              distributed=cc.get('fp16'),
                                                              enable_auto_augmentation='efficient' in cc.get(
                                                                  'network_name'))
    loss = nn.CrossEntropyLoss()
    #######################################
    # Search Working Device
    #######################################
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(working_device.type)
    print("-" * 100)
    ######################################
    # Set Model and Load
    #####################################
    nc = networks.NetworkQuantizationController(quantization_config=get_q_config_same(
        list(range(cc.get('activation_min_bit_width'), cc.get('activation_max_bit_width') + 1)),
        cc.get('bits_list'),
        cc.get('n_thresholds_shifts')), quantization_part=QUANTIZATION[cc.get('quantization_part')],
        ste=cc.get('gumbel_ste'))
    net = networks.get_network_function(cc.get('network_name'))(nc, pretrained=True)

    net = update_quantization_coefficient(net)
    param_out_list, activation_scale_params, variable_scale_params = model_coefficient_split(net)
    ######################################
    # Build Optimizer and Loss function
    #####################################
    optimizer = RAdam(
        [{'params': activation_scale_params, 'lr': cc.get('lr_activation'), 'weight_decay': 0.0},
         {'params': variable_scale_params, 'lr': cc.get('lr_coefficient'), 'weight_decay': 0.0}])
    optimizer_net = RAdam(
        [{'params': param_out_list, 'lr': cc.get('lr_start'), 'weight_decay': cc.get('weight_decay')}])
    net = net.to('cuda')
    if cc.get('fp16'):
        net, optimizers = amp.initialize(net, [optimizer, optimizer_net],
                                         opt_level='O1',
                                         keep_batchnorm_fp32=None,
                                         loss_scale=None
                                         )
    else:
        optimizers = [optimizer, optimizer_net]
    net = common.multiple_gpu_enable(net, apex=cc.get('fp16'))
    ##################################
    # Inital accuracy evalution
    ##################################
    test_base_acc = common.accuracy_evaluation(net, test_loader, working_device)
    print("Network Weight Loading Done with Accuracy:", test_base_acc)
    print('-' * 100)
    ######################################
    # Enable Quantization
    #####################################
    nc.apply_fix_point()
    #####################################
    # Search Max thresholds
    #####################################
    print("Initial thresholds", get_thresholds_list(nc, net)[0])
    nc.set_temperature(1)
    nc.enable_statistics_update()  # enable statistics collection
    train_acc = common.accuracy_evaluation(net, train_loader, working_device)
    nc.disable_statistics_update()  # disable statistics collection
    print("Initial Thresholds at the end of statistics update", get_thresholds_list(nc, net)[0], train_acc)
    #####################################
    # Retrain
    #####################################
    temp_func = common.get_exp_cycle_annealing(cc.get('cycle_size') * len(train_loader), cc.get('temp_step'),
                                               np.round(len(train_loader) / cc.get('n_gumbel')))
    gamma_target_func = common.get_step_annealing(cc.get('cycle_size'), CR_START, cc.get('target_compression'),
                                                  cc.get('n_target_steps'))
    gamma_target_func_activation = common.get_step_annealing(cc.get('cycle_size'), CR_START,
                                                             cc.get('target_compression_activation'),
                                                             cc.get('n_target_steps'))
    print("-" * 100)
    print("Starting Training")
    single_iteration_training_joint(net, cc, nc, train_loader, test_loader, optimizers, loss, temp_func,
                                    cc.get('gamma'), gamma_target_func,
                                    gamma_target_func_activation,
                                    working_device, amp_flag=cc.get('fp16'), train_sampler=train_sampler,
                                    gamma_rate=cc.get('gamma_rate'))
    final_stage_training(net, cc, nc, train_loader, test_loader, [optimizers[1]], loss,
                         working_device, cc.get('fp16'), train_sampler)


if __name__ == '__main__':
    base_runner()
