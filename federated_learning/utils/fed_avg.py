import torch

def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

    return new_params

def average_parameters(model_params, diffs_param):
    """
    Update new model params

    :param model_params: Old model parameters
    :type model_params: 
    :param diffs_params: new parameters from clients
    :type diffs_params: 
    """
    def avg(diff_avg, diff, num):
        new_avg = []
        for i, param in enumerate(diff_avg):
            new_avg.append((diff_avg[i] * num + diff[i]) / (num + 1))
        return new_avg

    diff_avg = diffs_param[0]
    for i, diff in enumerate(diffs_param[1]):
        diff_avg = avg(list(diff_avg), diff, torch.tensor([i + 1]))
    
    updated_model_params = [
        model_param - diff_param for model_param, diff_param in zip(model_params, diff_avg)
    ]

    return updated_model_params


