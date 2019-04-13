import math
from torch import nn
from torch.autograd import Function
import torch
import numpy as np
from skimage import measure

import crfloss_cpp

# torch.manual_seed(42)


class CRFLossFunction(Function):
    def __init__(self, sigma_xy=15.0, sigma_rgb=0.125):
        super(CRFLossFunction, self).__init__()
        self.sigma_xy = sigma_xy
        self.sigma_rgb = sigma_rgb

    def forward(self, input, image):
        loss = crfloss_cpp.forward(input, image.float(), self.sigma_xy, self.sigma_rgb)
        self.save_for_backward(input, image)
        return loss

    def backward(self, grad_output):
        input, image = self.saved_variables
        grad_input = crfloss_cpp.backward(grad_output, input, image.float(), self.sigma_xy, self.sigma_rgb)
        return grad_input, None


class CRFLoss(nn.Module):
    def __init__(self, sigma_xy=15.0, sigma_rgb=0.125):
        super(CRFLoss, self).__init__()
        self.sigma_xy = sigma_xy
        self.sigma_rgb = sigma_rgb

    def forward(self, input, image):
        output = CRFLossFunction(self.sigma_xy, self.sigma_rgb)(input, image)
        return output

    def _get_name(self):
        return 'CRFLoss'
