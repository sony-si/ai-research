import torch
from tqdm import tqdm
from datasets.data_prefercher import DataPreFetcher


def accuracy_factor(input_prediction, input_label):
    """
    The function return the number of correct classifications and the total number of labels
    :param input_prediction: Input prediction tensor of NxClass
    :param input_label: Input label tensor of size N
    :return: a tuple of floating point numbers (# of correct samples, # of total samples)
    """
    _, predicted = torch.max(input_prediction, 1)
    total = input_label.size(0)
    correct = (predicted == input_label.long()).sum().item()
    return correct, total


def accuracy_evaluation(input_net, dataset_loader, working_device):
    """
    The function return the top1 accuracy of a network
    :param input_net:Input network
    :param dataset_loader: Inputer dataset loader
    :param working_device: torch device
    :return: a floating point number of the Top1 accuracy
    """
    input_net = input_net.eval()
    correct_acc = 0
    total_acc = 0
    prefetcher = DataPreFetcher(dataset_loader)
    image, label = prefetcher.next()
    with tqdm(total=len(dataset_loader)) as pbar:
        while image is not None:
            pbar.update(1)
            if working_device.type == 'cuda':
                image = image.cuda()
                label = label.cuda()
            prediction = input_net(image)  # forward

            correct, total = accuracy_factor(prediction, label)
            correct_acc += correct
            total_acc += total
            image, label = prefetcher.next()
    return 100 * correct_acc / total_acc
