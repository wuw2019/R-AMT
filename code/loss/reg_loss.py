import torch
import torch.nn as nn
import torch.nn.functional as F

def RegLoss(param, k):
    assert k in [1,2]
    param = param.view(-1)
    reg_loss = torch.norm(param, k)
    return reg_loss


