from sklearn.preprocessing import StandardScaler
import torch

def apply_standard_scaler(gradients):
    scaler = StandardScaler()

    return scaler.fit_transform(gradients)

def scale_model(model, scale):
    """Scale the parameters of a model.
    Args:
        model (torch.nn.Module): the models whose parameters will be scaled.
        scale (float): the scaling factor.
    Returns:
        torch.nn.Module: the module with scaled parameters.
    """
    params = model.parameters()
    dict_params = dict(params)

    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)

    return model