import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from piq import vsi, haarpsi
import functools
import torch



def nrmse(images, recons):
    score = 0
    num_slices = images.shape[0]
    for i in range(num_slices):
        image = images[i]
        recon = recons[i]
        # calculate normalized mean squared error
        # return np.linalg.norm(image-recon)**2 / np.linalg.norm(image)**2
        dim1 = image.shape[0]
        dim2 = image.shape[1]
        score += np.sqrt(np.sum((image-recon)**2)/(dim1*dim2)) / np.mean(image)
    return score / num_slices
def nmse(gt, pred):
    return np.linalg.norm(gt-pred)**2/np.linalg.norm(gt)**2
def ssim(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if gt.ndim != pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")
    maxval = np.max(gt) if maxval is None else maxval
    _ssim = sum(
        structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval) for slice_num in range(gt.shape[0])
    )
    return _ssim / gt.shape[0]


def haarpsi(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if gt.ndim != 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if gt.ndim != pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")
    reduction= 'mean'
    scales = 3
    subsample= True
    c= 30.0
    alpha = 4.2

    maxval = np.max(gt) if maxval is None else maxval
    _haarpsi = functools.partial(haarpsi, scales=scales, subsample=subsample, c=c, alpha=alpha,
                                     data_range=maxval, reduction=reduction)
    __haarpsi = sum(
       _haarpsi(torch.from_numpy(gt[slice_num]).unsqueeze(0).unsqueeze(0).float(), torch.from_numpy(pred[slice_num]).unsqueeze(0).unsqueeze(0).float()) for slice_num in range(gt.shape[0])
    ).numpy()

    return __haarpsi / gt.shape[0]

def vsi(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if gt.ndim != 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if gt.ndim != pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")
    reduction= 'mean'
    c1: float = 1.27
    c2: float = 386.
    c3: float = 130.
    alpha: float = 0.4
    beta: float = 0.02
    data_range:[int, float] = 1.
    omega_0: float = 0.021
    sigma_f: float = 1.34
    sigma_d: float = 145.
    sigma_c: float = 0.001

    maxval = np.max(gt) if maxval is None else maxval
    _vsi = functools.partial(
        vsi, c1=c1, c2=c2, c3=c3, alpha=alpha, beta=beta, omega_0=omega_0,
        sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c, data_range=maxval,
        reduction=reduction)
    __vsi = sum(
       _vsi(torch.from_numpy(gt[slice_num]).unsqueeze(0).unsqueeze(0).float(), torch.from_numpy(pred[slice_num]).unsqueeze(0).unsqueeze(0).float()) for slice_num in range(gt.shape[0])
    )

    return __vsi / gt.shape[0]

