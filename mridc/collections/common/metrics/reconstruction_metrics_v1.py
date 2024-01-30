# coding=utf-8
__author__ = "Dimitrios Karkalousos"

# Parts of the code have been taken from https://github.com/facebookresearch/fastMRI

import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from piq import vsi, haarpsi
import torch
import functools

def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)  # type: ignore


def nmse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = np.max(gt)
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if gt.ndim != 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if gt.ndim != pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = np.max(gt) if maxval is None else maxval

    _ssim = sum(
        structural_similarity(gt[slice_num], pred[slice_num], data_range=maxval) for slice_num in range(gt.shape[0])
    )

    return _ssim / gt.shape[0]

def haarpsi3d(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
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

def vsi3d(gt: np.ndarray, pred: np.ndarray, maxval: np.ndarray = None) -> float:
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


METRIC_FUNCS = dict(MSE=mse, NMSE=nmse, PSNR=psnr, SSIM=ssim,HaarPSI=haarpsi3d,VSI=vsi3d)


class Metrics:
    """Maintains running statistics for a given collection of metrics."""

    def __init__(self, metric_funcs, output_path, method):
        """
        Parameters
        ----------
        metric_funcs (dict): A dict where the keys are metric names and the values are Python functions for evaluating
        that metric.
        output_path: path to the output directory
        method: reconstruction method
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}
        self.output_path = output_path
        self.method = method

    def push(self, target, recons):
        """
        Pushes a new batch of metrics to the running statistics.

        Parameters
        ----------
        target: target image
        recons: reconstructed image

        Returns
        -------
        dict: A dict where the keys are metric names and the values are
        """
        for metric, func in METRIC_FUNCS.items():
            self.metrics_scores[metric].push(func(target, recons))

    def means(self):
        """Mean of the means of each metric."""
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """Standard deviation of the means of each metric."""
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """Representation of the metrics."""
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = " ".join(f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}" for name in metric_names) + "\n"

        with open(f"{self.output_path}metrics.txt", "a") as output:
            output.write(f"{self.method}: {res}")

        return res
