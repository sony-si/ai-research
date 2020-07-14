import common
import torch

from tqdm import tqdm
from apex import amp
from quantization_common.quantization_information import calculate_expected_weight_compression
from datasets.data_prefercher import DataPreFetcher


def batch_step(pbar, net, image, optimizers, label, criterion, gamma, gamma_target, gamma_rate, amp_flag,
               working_device):
    """
    This function execute a batch step
    :param pbar: Progress bar object
    :param net: Input network module
    :param image: batch of images
    :param optimizers: A list of optimizers
    :param label: array of labels
    :param criterion: The loss criterion
    :param gamma: the of weights loss term
    :param gamma_target: the target weights compression
    :param gamma_rate:  the target weights compression power factor
    :param amp_flag: current working device
    :param working_device: using amp boolean
    :return: None
    """
    pbar.update(1)
    image = image.to(working_device)
    label = label.to(working_device)
    prediction = net(image)  # forward
    correct, total = common.accuracy_factor(prediction,
                                            label)  # calculate accuracy factor # of correct examples
    l = criterion(prediction, label)  # loss function
    r = calculate_expected_weight_compression(net)
    if gamma > 0.0:
        l = l + gamma * torch.pow(torch.relu((gamma_target - r) / gamma_target), gamma_rate)

    if amp_flag:
        with amp.scale_loss(l, optimizers) as scaled_loss:
            scaled_loss.backward()
    else:
        l.backward()
    [op.step() for op in optimizers]  # optimize all params

    [op.zero_grad() for op in optimizers]  # optimize all params
    return correct, total, l.item()


def batch_loop(net, nc, optimizers, train_loader, working_device, criterion, amp_flag, gamma, gamma_rate,
               temp_func=None, gamma_target=0.0, epoch=0):
    """

    :param net: Input network module
    :param nc: Network Config class
    :param optimizers:  a list of optimizers
    :param train_loader: The training dataset loader
    :param working_device: using amp boolean
    :param criterion: The loss criterion
    :param amp_flag: current working device
    :param gamma: the of weights loss term
    :param gamma_rate:  the target weights compression power factor
    :param temp_func: The gumbel softmax temperature function
    :param gamma_target: the target weights compression
    :param epoch: Epoch index
    :return: None
    """
    n = len(train_loader)
    i = 0
    t = None
    loss_meter = common.AverageMeter()
    accuracy_meter = common.AverageMeter()

    with tqdm(total=n) as pbar:
        prefetcher = DataPreFetcher(train_loader)
        image, label = prefetcher.next()
        while image is not None:
            if temp_func is not None:
                t = temp_func(epoch * n + i)
                nc.set_temperature(t)
            correct, total, loss_value = batch_step(pbar, net, image, optimizers, label, criterion, gamma,
                                                    gamma_target,
                                                    gamma_rate, amp_flag, working_device)
            loss_meter.update(loss_value)
            accuracy_meter.update(correct, n=total)
            i += 1
            image, label = prefetcher.next()
    torch.cuda.synchronize()
    return loss_meter.avg, 100 * accuracy_meter.avg, t
