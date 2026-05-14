import torch

def freeze_parameters(model: torch.nn.Module):
    """
    Freeze the parameters of a model.
    """
    for param in model.parameters():
        param.requires_grad = False
    
    return model