import torch


class ParallelFake(torch.nn.Module):
    def __init__(self, module):
        super(ParallelFake, self).__init__()
        self.module = module
        self.add_module('module', module)

    def forward(self, x_in):
        return self.module(x_in)


def multiple_gpu_enable(input_net, apex=False):
    """
    This function return a torch module for multiple GPU usage, If there not GPU on the PC then a ParallelFake is
    used to keep the same naming.
    :param input_net: Input torch module
    :param apex: boolean flag indication if the GPU data parallel is base on apex
    :return: A torch module
    """
    if torch.cuda.is_available():
        if apex:
            from apex.parallel import DistributedDataParallel as DDP
            net = DDP(input_net, delay_allreduce=True)
        else:
            net = torch.nn.DataParallel(input_net).cuda()
    else:
        net = ParallelFake(input_net)
    return net
