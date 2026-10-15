
import numpy as np
from skimage import exposure, filters, morphology, feature

def normalize(img, mode='minmax'):
    """影像正規化"""
    img = img.astype(np.float64)

    if mode == 'minmax':
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img


def denoise(img, method='median'):
    """去噪處理，支援 2D 與 3D 影像"""
    if method == 'median':
        if img.ndim == 2:
            return filters.median(img, morphology.disk(2))
        elif img.ndim == 3:
            return filters.median(img, morphology.ball(2))

    elif method == 'gaussian':
        return filters.gaussian(img, sigma=1.0)

    return img


def apply_clahe(img, clip_limit=0.03):
    """自適應直方圖等化"""
    return exposure.equalize_adapthist(img, clip_limit=clip_limit)


def apply_gamma(img, gamma=1.0):
    """Gamma 校正"""
    return exposure.adjust_gamma(img, gamma=gamma)


def detect_edges(img, method='sobel'):
    """邊緣檢測"""
    if method == 'canny':
        return feature.canny(img)

    return filters.sobel(img)


def augment(img, flip=True, rot=True):
    """資料擴增"""
    out = [img]

    if flip:
        out += [
            np.flip(img, axis=-1),
            np.flip(img, axis=-2)
        ]

    if rot:
        out += [
            np.rot90(img, k, axes=(-2, -1))
            for k in (1, 2, 3)
        ]

    return out
