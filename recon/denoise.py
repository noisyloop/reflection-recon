"""
Wavelet-based denoiser for PRNU residual extraction.

Implements the Mihcak MMSE (local-Wiener) estimator used in the canonical
Lukas-Fridrich-Goljan PRNU pipeline. The noise-free image estimate F(I) is
obtained by shrinking wavelet detail coefficients toward zero based on a
per-coefficient local variance estimate; the PRNU-bearing residual is then
W = I - F(I).

This replaces the Gaussian high-pass used in the lightweight browser build.
"""
import numpy as np
import pywt


def _mmse_shrink(coeff, var_noise):
    """Mihcak local-variance MMSE shrink of one detail subband.

    The local signal variance is estimated with several window sizes and the
    minimum is taken (conservative), then a Wiener gain is applied.
    """
    est = None
    for w in (3, 5, 7, 9):
        k = np.ones((w, w), dtype=np.float64) / (w * w)
        # local mean of squared coefficients via separable box filter
        local = _box(coeff * coeff, w)
        v = np.maximum(0.0, local - var_noise)
        est = v if est is None else np.minimum(est, v)
    return coeff * est / (est + var_noise)


def _box(a, w):
    """Separable box filter with symmetric padding."""
    r = w // 2
    ap = np.pad(a, r, mode="reflect")
    # rows
    c = np.cumsum(ap, axis=0)
    c = np.vstack([c[w - 1:w, :], c[w:, :] - c[:-w, :]])
    # cols
    c = np.cumsum(c, axis=1)
    c = np.hstack([c[:, w - 1:w], c[:, w:] - c[:, :-w]])
    return c[:a.shape[0], :a.shape[1]] / (w * w)


def wavelet_denoise(img, sigma=3.0, wavelet="db8", levels=4):
    """Return F(I): the denoised (noise-free) estimate of a single channel.

    Parameters
    ----------
    img : 2-D float array
    sigma : assumed sensor-noise std in the same units as ``img`` (8-bit -> ~2-5)
    wavelet, levels : DWT parameters
    """
    img = np.asarray(img, dtype=np.float64)
    var_noise = float(sigma) ** 2
    coeffs = pywt.wavedec2(img, wavelet, level=levels, mode="periodization")
    out = [coeffs[0]]
    for (cH, cV, cD) in coeffs[1:]:
        out.append((
            _mmse_shrink(cH, var_noise),
            _mmse_shrink(cV, var_noise),
            _mmse_shrink(cD, var_noise),
        ))
    den = pywt.waverec2(out, wavelet, mode="periodization")
    return den[:img.shape[0], :img.shape[1]]


def noise_residual(img, sigma=3.0, wavelet="db8", levels=4):
    """W = I - F(I). The PRNU-bearing noise residual of one channel."""
    img = np.asarray(img, dtype=np.float64)
    return img - wavelet_denoise(img, sigma, wavelet, levels)
