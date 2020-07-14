import torch
from torch import nn
from torch.distributions.uniform import Uniform


class GumbelSoftmax(nn.Module):
    def __init__(self, ste=False, log_softmax_enable: bool = True, eps: float = 1e-6):
        """
        A Gumbel Softmax module
        :param ste: boolean flag stating if STE is enabled
        :param log_softmax_enable: The gumbel softmax input go through log softmax function
        :param eps: The gumbel softmax epsilon
        """
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.u = Uniform(0, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax_enable = log_softmax_enable
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.ste = ste

    def forward(self, pi, temperature, batch_size: int = 1, noise_disable: float = False):
        """
        The forward function of the Gumbel Softmax module

        :param pi: Input tensor
        :param temperature: The gumbel softmax temperature
        :param batch_size: the number of batch for the same pi input
        :param  noise_disable: boolean that disables the gumbel noise
        :return: A tensor after gumbel softmax
        """
        g = torch.zeros([batch_size, *[i for i in pi.shape]])
        t = 1
        if not noise_disable:
            g = - torch.log(
                -torch.log(
                    self.u.sample(
                        [batch_size, *[i for i in pi.shape]]) + self.eps) + self.eps)  # Gumbul Softmax
            t = temperature
        if pi.is_cuda:
            g = g.cuda()
        search_matrix = pi.reshape(1, -1)
        if self.log_softmax_enable:
            search_matrix = self.log_softmax(search_matrix)
        g = g.reshape(batch_size, -1)
        s = (search_matrix + g) / t
        p = self.softmax(s).reshape([batch_size, *pi.shape])
        if self.ste:
            p_flatten = p.reshape([batch_size, -1])
            p_onehot = torch.FloatTensor(batch_size, p_flatten.shape[1]).cuda()
            p_onehot.zero_()
            p_onehot.scatter_(1, p_flatten.argmax(dim=-1).reshape([-1, 1]), 1)
            p_onehot = p_onehot.reshape([batch_size, *pi.shape])
            error = (p_onehot - p).detach()
            p = p + error
        return p
