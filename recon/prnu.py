"""
PRNU sensor-fingerprint estimation and matching.

Pipeline (Lukas, Fridrich, Goljan 2006; Chen/Fridrich/Goljan/Lukas 2008):

  1. residual        W_i = I_i - F(I_i)          (wavelet denoiser)
  2. ML fingerprint  K_hat = sum(W_i * I_i) / sum(I_i^2)
  3. post-process    zero-mean rows/cols  +  Wiener-in-DFT
  4. detection       correlate a test residual W against the predicted
                     signal (I * K_hat); report NCC and FFT-based PCE.

PCE (peak-to-correlation-energy) is computed over the *full* 2-D circular
cross-correlation surface via the FFT, not a decoy-shift approximation.
"""
import numpy as np

from .denoise import noise_residual


# ---------------------------------------------------------------- fingerprint
def estimate_fingerprint(images, sigma=3.0, wavelet="db8", levels=4):
    """Maximum-likelihood PRNU estimate K_hat from flat, bright frames.

    ``images`` : iterable of 2-D float arrays (single channel), same shape.
    """
    num = None
    den = None
    for img in images:
        img = np.asarray(img, dtype=np.float64)
        w = noise_residual(img, sigma, wavelet, levels)
        if num is None:
            num = np.zeros_like(img)
            den = np.zeros_like(img)
        num += w * img
        den += img * img
    if num is None:
        raise ValueError("no images supplied")
    K = num / (den + 1.0)
    return postprocess(K)


def postprocess(K):
    """Zero-mean rows/cols then Wiener-in-DFT to strip non-unique artifacts."""
    K = zero_mean_total(K)
    K = wiener_dft(K)
    return K


def zero_mean_total(K):
    """Remove row and column means (kills shared CFA / linear pattern)."""
    K = K - K.mean(axis=1, keepdims=True)
    K = K - K.mean(axis=0, keepdims=True)
    return K


def wiener_dft(K, sigma=None):
    """Suppress periodic artifacts by Wiener-filtering the magnitude spectrum."""
    F = np.fft.fft2(K)
    mag = np.abs(F)
    if sigma is None:
        sigma = np.sqrt(np.mean(mag ** 2)) * 0.5
    mag_clean = mag ** 2 / (mag ** 2 + sigma ** 2) * mag
    # avoid division by zero, keep phase
    scale = np.divide(mag_clean, mag, out=np.zeros_like(mag), where=mag > 1e-9)
    return np.real(np.fft.ifft2(F * scale))


# ---------------------------------------------------------------- detection
def _zero_mean(a):
    return a - a.mean()


def ncc(a, b):
    """Normalized cross-correlation at zero shift (a single scalar)."""
    a = _zero_mean(np.asarray(a, dtype=np.float64))
    b = _zero_mean(np.asarray(b, dtype=np.float64))
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 1e-12 else 0.0


def crosscorr(a, b):
    """Full 2-D circular cross-correlation surface via FFT."""
    a = _zero_mean(np.asarray(a, dtype=np.float64))
    b = _zero_mean(np.asarray(b, dtype=np.float64))
    C = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    return C


def pce(a, b, squared_radius=1):
    """FFT-based peak-to-correlation-energy (Goljan).

    Returns (pce_value, peak_yx). The correlation is normalized to unit energy,
    so PCE = peak^2 / (mean energy away from an exclusion neighborhood), signed
    by the peak. PCE > ~50 is a confident same-sensor match.
    """
    C = crosscorr(a, b)
    n = C.size
    # normalize so total energy is comparable across images
    C = C / np.sqrt(np.sum(C ** 2) / n + 1e-30)
    peak = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    peak_val = C[peak]
    # exclusion neighborhood around the peak (toroidal)
    mask = np.ones_like(C, dtype=bool)
    yy, xx = peak
    r = squared_radius
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            mask[(yy + dy) % C.shape[0], (xx + dx) % C.shape[1]] = False
    energy = np.sum(C[mask] ** 2) / mask.sum()
    val = float(peak_val ** 2 / energy * np.sign(peak_val)) if energy > 1e-30 else 0.0
    return val, (int(peak[0]), int(peak[1]))


def verify(test_img, K, sigma=3.0, wavelet="db8", levels=4):
    """Test whether ``test_img`` came from the sensor with fingerprint ``K``.

    Correlates the image residual W against the predicted PRNU signal I*K.
    Returns dict with ncc, pce, peak.
    """
    img = np.asarray(test_img, dtype=np.float64)
    W = noise_residual(img, sigma, wavelet, levels)
    pred = img * K
    p, peak = pce(W, pred)
    return {"ncc": ncc(W, pred), "pce": p, "peak": peak}
