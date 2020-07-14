import common
import time
import torch
import wandb
import os

from common.parto_best import PartoBest

from quantization_common.quantization_information import calculate_weight_compression, \
    calculate_activation_max_compression
from quantization_common.quantization_activation import update_network_activation
from training.training_functions import batch_loop


def single_iteration_training_joint(net, cc, nc, train_loader, test_loader, optimizers, loss, temp_func,
                                    gamma, gamma_target_func,
                                    gamma_target_func_activation,
                                    working_device, amp_flag=False, train_sampler=None, gamma_rate=2.0):
    """
    The joint training of HMQ parameters and network parameters
    :param net: Input network module
    :param cc: argument dictionary
    :param nc: Network Config class
    :param train_loader: The training dataset loader
    :param test_loader: The test dataset loader
    :param optimizers: a list of optimizers
    :param loss: The task loss function
    :param temp_func: The gumbel softmax temperature function
    :param gamma: the of weights loss term
    :param gamma_target_func: the target weights compression function
    :param gamma_target_func_activation: the activation weights compression function
    :param working_device: current working device
    :param amp_flag: using amp boolean
    :param train_sampler: training dataset sampler
    :param gamma_rate: the target weights compression power factor
    :return: None
    """
    print("Start Search Training Stage")
    pb = PartoBest()
    for e in range(cc.get('n_epochs')):
        if amp_flag and train_sampler is not None:
            train_sampler.set_epoch(e)
        gamma_target = gamma_target_func(e)

        update_network_activation(nc, net, gamma_target_func_activation(e))
        s = time.time()

        if e == 0:  # Measure Inital accuracy
            print("Run Initial Results of Validation & Training Sets")
            nc.set_temperature(1)
            val_acc = common.accuracy_evaluation(net, test_loader, working_device)
            train_acc = common.accuracy_evaluation(net, train_loader, working_device)
            if cc.get('local_rank') == 0:
                wc = calculate_weight_compression(net)
                ac = calculate_activation_max_compression(net)
                lr = optimizers[0].param_groups[0]['lr']
                wandb.log(
                    {'Validation Accuracy': val_acc, 'Training Accuracy': train_acc, 'Compression Rate': wc,
                     'Compression Max Rate Activation': ac,
                     'lr': lr})
            print("Initial Validation & Training Result:", val_acc, train_acc)
            print("-" * 100)
        print("Start Epoch:", e)
        net = net.train()
        [op.zero_grad() for op in optimizers]  # zero the parameter gradients

        loss_value, train_acc, temp = batch_loop(net, nc, optimizers, train_loader, working_device, loss, amp_flag,
                                                 gamma, gamma_rate,
                                                 temp_func,
                                                 gamma_target=gamma_target, epoch=e)
        print("Start Validation Run")

        val_acc = common.accuracy_evaluation(net, test_loader, working_device)

        if cc.get('local_rank') == 0:
            lr = optimizers[0].param_groups[0]['lr']
            wc = calculate_weight_compression(net)
            ac = calculate_activation_max_compression(net)
            wandb.log(
                {'Loss': loss_value, 'Validation Accuracy': val_acc, 'Training Accuracy': train_acc,
                 'Compression Rate': wc,
                 'Compression Max Rate Activation': ac,
                 'Temperature': temp,
                 'lr': lr})
        ###############################
        # Save Model & Result
        ###############################
        if cc.get('local_rank') == 0:
            if pb.is_pareto_best(wc, val_acc):
                torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'parto_best.pt'))  # TODO: log generation
            torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'last.pt'))
            print('End Epoch:', e, ' Run Time:', time.time() - s, ' Validation Acc:', val_acc, " Training Acc:",
                  train_acc, " Compression Rate:",
                  wc, " Compression Rate Activation:",
                  ac)
            print("-" * 100)
    print("End Search")
