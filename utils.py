import torch
import torch.nn as nn

def update_momentum(model: nn.Module, model_ema: nn.Module, m: float): #taken from lightly https://github.com/lightly-ai/lightly/blob/4685d72f3b69366c614f8ccf4db122b7bb4efb8c/lightly/models/utils.py#L186
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`
    Momentum encoders are a crucial component fo models such as MoCo or BYOL. 
    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    model.eval()
    model_ema.eval()
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1. - m)
