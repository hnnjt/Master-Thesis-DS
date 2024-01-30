# coding=utf-8
__author__ = "Tim Paquaij"

# Parts of the code have been taken from https://github.com/photosynthesis-team/piq
import torch
import torch.nn as nn
from piq.haarpsi import haarpsi
import functools

class HaarPSILoss(nn.Module):
    r"""Creates a criterion that measures  Haar Wavelet-Based Perceptual Similarity loss between
    each element in the input and target.

    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        scales: Number of Haar wavelets used for image decomposition.
        subsample: Flag to apply average pooling before HaarPSI computation. See references for details.
        c: Constant from the paper. See references for details
        alpha: Exponent used for similarity maps weightning. See references for details

    Examples:

       # >>> loss = HaarPSILoss()
        #>>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        #>>> y = torch.rand(3, 3, 256, 256)
        #>>> output = loss(x, y)
        #>>> output.backward()

    References:
        R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand (2017)
        'A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment'
        http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf
    """
    def __init__(self, reduction: [str] = 'mean',
                 scales: int = 3, subsample: bool = True, c: float = 30.0, alpha: float = 4.2) -> None:
        super().__init__()
        self.reduction = reduction

        self.reduction=reduction
        self.scales =scales
        self.subsample=subsample
        self.c =c
        self.alpha =alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor,data_range: torch.tensor):
        r"""Computation of HaarPSI as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of HaarPSI loss to be minimized in [0, 1] range.
        """
        data_range = data_range[:, None, None, None]
        self.haarpsi = functools.partial(haarpsi, scales=self.scales, subsample=self.subsample, c=self.c, alpha=self.alpha,data_range=data_range, reduction=self.reduction)
        return 1.-self.haarpsi(x=x, y=y)