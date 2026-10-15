
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def psnr(img1, img2, data_range=1.0):
    """計算 PSNR"""
    return peak_signal_noise_ratio(
        img1, img2, data_range=data_range
    )


def ssim(img1, img2, data_range=1.0):
    """計算 SSIM"""
    return structural_similarity(
        img1,
        img2,
        data_range=data_range
    )
