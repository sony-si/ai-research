from networks.mobilenet_v2 import mobilenet_v2
from networks.resnet import resnet50
from networks.mobilenet_v1 import mobilenet_v1
from networks.efficient_net import efficient_net

network_dict = {'mobilenet_v1': mobilenet_v1,
                'mobilenet_v2': mobilenet_v2,
                'efficient_net': efficient_net,
                'resnet50': resnet50}


def get_network_function(net_name):
    net = network_dict.get(net_name)
    if net is None:
        raise Exception('Cant find network named:' + net_name)
    return net
