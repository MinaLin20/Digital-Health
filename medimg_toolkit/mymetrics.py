
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


def auroc(y_true, score):
    """計算二分類 AUROC（依 A5 陽性分數 > 陰性分數的配對比例計算，平手計 0.5）"""
    y_true = np.asarray(y_true).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()

    pos = score[y_true == 1]
    neg = score[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    diff = pos[:, None] - neg[None, :]
    concordant = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)

    return float(concordant / (len(pos) * len(neg)))
