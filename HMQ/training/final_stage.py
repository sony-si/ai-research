import common
import time
import torch
import wandb
import os

from common.single_best import IsBest

from quantization_common.quantization_information import calculate_weight_compression, \
    calculate_activation_max_compression
from training.training_functions import batch_loop


def final_stage_training(net, cc, nc, train_loader, test_loader, optimizers, criterion,
                         working_device, amp_flag, train_sampler):
    """
    The final stage training function:
        Loop epoch:
            batch_loop_function
    :param net: Input network module
    :param cc: argument dictionary
    :param nc: Network Config class
    :param train_loader: The training dataset loader
    :param test_loader: The test dataset loader
    :param optimizers: a list of optimizers
    :param criterion: The loss criterion
    :param working_device: current working device
    :param amp_flag: using amp boolean
    :param train_sampler: training dataset sampler
    :return: None
    """
    print("Start Final Training Stage")
    nc.set_temperature(-1)
    best = IsBest()
    # Loop Epochs
    for e in range(cc.get('n_epochs_final')):
        s = time.time()
        if amp_flag and train_sampler is not None:
            train_sampler.set_epoch(e)
        net = net.train()
        [op.zero_grad() for op in optimizers]  # zero the parameter gradients

        loss_value, train_acc, t = batch_loop(net, nc, optimizers, train_loader, working_device, criterion, amp_flag,
                                              gamma=0,
                                              gamma_rate=1, temp_func=None, gamma_target=1)
        print("Start Validation Run")
        val_acc = common.accuracy_evaluation(net, test_loader, working_device)
        if cc.get('local_rank') == 0:
            wandb.log(
                {'Loss': loss_value, 'Validation Accuracy': val_acc, 'Training Accuracy': train_acc,
                 'Compression Rate': calculate_weight_compression(net),
                 'Compression Max Rate Activation': calculate_activation_max_compression(net),
                 'lr': optimizers[0].param_groups[0]['lr']})

        if cc.get('local_rank') == 0:
            if best.is_best(val_acc):
                torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'final_best.pt'))
            torch.save(net.state_dict(), os.path.join(wandb.run.dir, 'final_last.pt'))
            print('End Epoch:', e, ' Run Time:', time.time() - s, ' Validation Acc:', val_acc, " Training Acc:",
                  train_acc)
            print("-" * 100)
    print("End Final Training Loop")
