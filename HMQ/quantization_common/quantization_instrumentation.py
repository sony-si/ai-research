from networks import layers


def model_coefficient_split(model):
    """
    This function split the model weights into 3 groups:
        1) the original module weights
        2) Activation HMQ weights
        3) Weights HMQ weights
    :param model: Input a PyTorch model wit HMQ Block
    :return: a tuple of weights: 1) the original module weights 2) Activation HMQ weights 3) Weights HMQ weights
    """
    quantization_nodes = [m for m in model.modules() if isinstance(m, layers.Quantization)]
    nl_nodes = [m for m in model.modules() if isinstance(m, layers.NonLinear)]
    conv_bn_nodes = [m for m in model.modules() if isinstance(m, layers.ConvBN) or isinstance(m, layers.FullyConnected)]
    param_list = list(model.parameters())
    n_params = len(param_list)
    if len(quantization_nodes) != (len(nl_nodes) + len(conv_bn_nodes)):
        raise Exception('Mismatch between the number of quantization node and operation nodes')

    # activation
    activation_list = []
    for n in nl_nodes:  # loop all nodes
        for p in n.q.parameters():  # loop quantization parameters
            activation_list.append(p)

    # activation
    variable_list = []
    for n in conv_bn_nodes:  # loop all nodes
        for p in n.q.parameters():  # loop quantization parameters
            variable_list.append(p)

    param_out_list = [p for p in param_list if
                      not any([p is a for a in activation_list]) and not any([p is a for a in variable_list])]
    print(n_params, len(param_out_list), len(activation_list), len(variable_list))
    if n_params != (len(param_out_list) + len(activation_list) + len(variable_list)):
        raise Exception('Mismatch between the number of params')
    return param_out_list, activation_list, variable_list


def update_quantization_coefficient(model):
    """
    Update quantization nodes parameter according to quantization config
    :param model: Input PyTorch Module
    :return: A PyTorch Module after initialization of quantization coefficients.
    """
    print("Add Quantization Coefficient")
    for n, m in model.named_modules():
        if isinstance(m, layers.Quantization):
            m.init_quantization_coefficients()
    return model
