"""
Corneal point-spread-function estimation.

The catchlight (Purkinje reflection) in an eye is the image of a near-point
light source. Its captured shape is therefore a direct sample of the system PSF
(defocus + diffraction + motion) at that location. Extracting and normalizing it
gives a *physically grounded* PSF for Richardson-Lucy deconvolution, instead of
assuming a fixed Gaussian.

Falls back to a defocus-disc PSF when no clean catchlight is present.
"""
import numpy as np


def disc_psf(radius):
    """Normalized defocus (pillbox) PSF of given radius in pixels."""
    r = int(np.ceil(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    psf = ((x * x + y * y) <= radius * radius).astype(np.float64)
    s = psf.sum()
    return psf / s if s else psf


def gaussian_psf(sigma, size=None):
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    psf = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return psf / psf.sum()


def estimate_psf(eye_gray, size=15, thresh_pct=92.0):
    """Estimate the PSF from the brightest catchlight blob.

    Parameters
    ----------
    eye_gray : 2-D float array (single-channel eye/ROI crop)
    size     : odd side length of the returned PSF kernel
    thresh_pct : percentile used to background-subtract the highlight

    Returns
    -------
    psf : normalized (sum=1) 2-D kernel of shape (size, size)
    """
    img = np.asarray(eye_gray, dtype=np.float64)
    if size % 2 == 0:
        size += 1
    r = size // 2

    # locate the brightest pixel (the catchlight)
    yx = np.unravel_index(np.argmax(img), img.shape)
    y0 = int(np.clip(yx[0], r, img.shape[0] - r - 1))
    x0 = int(np.clip(yx[1], r, img.shape[1] - r - 1))
    patch = img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].copy()

    # background subtract + threshold to isolate the highlight energy
    bg = np.percentile(patch, thresh_pct)
    patch = np.clip(patch - bg, 0, None)
    s = patch.sum()
    if s < 1e-6:
        # no usable highlight -> fall back to a small defocus disc
        return disc_psf(max(1.0, size / 6.0))
    return patch / s
