"""ALL code from: https://github.com/wdika/mridc/blob/main/mridc/collections/common/part/utils.py"""
import torch


def complex_mul(x, y):
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Parameters
    ----------
    x: A PyTorch tensor with the last dimension of size 2.
    y: A PyTorch tensor with the last dimension of size 2.

    Returns
    -------
    A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x):
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Parameters
    ----------
    x: A PyTorch tensor with the last dimension of size 2.

    Returns
    -------
    A PyTorch tensor with the last dimension of size 2.
    """
    if x.shape[-1] != 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def sense(data, sensitivity_maps, dim: int = 0):
    """
    FROM MRIDC
    The SENSitivity Encoding (SENSE) transform [1]_.

    References
    ----------
    .. [1] Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: Sensitivity encoding for fast MRI. Magn Reson Med 1999; 42:952-962.

    Parameters
    ----------
    data: The input tensor
    sensitivity_maps: The sensitivity maps
    dim: The coil dimension

    Returns
    -------
    A coil-combined image.
    """
    data = torch.from_numpy(data)
    data = torch.view_as_real(data)

    sensitivity_maps = torch.from_numpy(sensitivity_maps)
    sensitivity_maps = torch.view_as_real(sensitivity_maps)
    recons1 = complex_mul(data, complex_conj(sensitivity_maps))
    recons = recons1.sum(0)
    return torch.view_as_complex(recons).numpy()

