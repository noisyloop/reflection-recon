"""
Image registration for shift-and-add frame stacking.

  * translation      : subpixel phase cross-correlation
  * rotation + scale  : Fourier-Mellin (log-polar of the FFT magnitude)

The rotation/scale path follows Reddy & Chatterji (1996): rotation and scale
become translations in the log-polar transform of the Fourier magnitude, which
is itself shift-invariant. We recover (angle, scale) there, undo them, then
recover the residual translation.
"""
import numpy as np
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale


def register_translation(ref, mov, upsample=10):
    """Return (dy, dx) sub-pixel shift that maps ``mov`` onto ``ref``."""
    shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample)
    return float(shift[0]), float(shift[1])


def _rc_highpass(shape):
    """Reddy-Chatterji high-pass emphasis H = (1-X)(2-X), X=cos(pi u)cos(pi v).

    Suppresses the DC blob so mid-frequency structure drives the log-polar
    match; without it, smooth images give an ambiguous rotation/scale estimate.
    """
    y = np.linspace(-0.5, 0.5, shape[0])[:, None]
    x = np.linspace(-0.5, 0.5, shape[1])[None, :]
    X = np.cos(np.pi * y) * np.cos(np.pi * x)
    return (1.0 - X) * (2.0 - X)


def register_rst(ref, mov, upsample=10):
    """Recover rotation (deg), scale, and (dy,dx) translation of ``mov``.

    Returns dict {angle, scale, shift}. Assumes square, same-shape inputs for
    the log-polar step (crop/pad beforehand if needed).
    """
    ref = np.asarray(ref, dtype=np.float64)
    mov = np.asarray(mov, dtype=np.float64)

    # 1. FFT magnitudes (shift-invariant), windowed
    win = np.hanning(ref.shape[0])[:, None] * np.hanning(ref.shape[1])[None, :]
    hp = _rc_highpass(ref.shape)
    f_ref = np.abs(np.fft.fftshift(np.fft.fft2(ref * win))) * hp
    f_mov = np.abs(np.fft.fftshift(np.fft.fft2(mov * win))) * hp

    # 2. log-polar transform -> rotation is a shift in angle, scale in log-r
    radius = min(ref.shape) // 2
    lp_ref = warp_polar(f_ref, radius=radius, scaling="log", order=1)
    lp_mov = warp_polar(f_mov, radius=radius, scaling="log", order=1)

    # 3. phase-correlate the log-polar images -> |angle|, scale
    shift, _, _ = phase_cross_correlation(lp_ref, lp_mov, upsample_factor=upsample)
    d_angle, d_logr = shift[0], shift[1]

    a0 = (360.0 / lp_ref.shape[0]) * d_angle
    klog = np.log(radius) / lp_ref.shape[1]
    scale = float(np.exp(d_logr * klog))

    # 4. resolve the sign / 180-deg ambiguity: try candidates, keep best fit.
    def _corrected(angle):
        c = rotate(mov, -angle, order=1, preserve_range=True)
        if abs(scale - 1.0) > 1e-3:
            c = rescale(c, scale, order=1, preserve_range=True, anti_aliasing=True)
            c = _fit(c, ref.shape)
        return c

    best = None
    for cand in (a0, -a0, a0 + 180.0, -a0 + 180.0):
        c = _corrected(cand)
        dy, dx = register_translation(ref, c, upsample)
        aligned = apply_shift(c, dy, dx)
        score = _ncc(ref, aligned)
        if best is None or score > best[0]:
            best = (score, cand, (dy, dx))

    _, angle, sh = best
    angle = ((angle + 180.0) % 360.0) - 180.0             # wrap to (-180, 180]
    return {"angle": float(angle), "scale": scale, "shift": sh}


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 1e-12 else 0.0


def _fit(a, shape):
    """Center-crop or pad ``a`` to ``shape``."""
    out = np.zeros(shape, dtype=a.dtype)
    h = min(a.shape[0], shape[0])
    w = min(a.shape[1], shape[1])
    sy = (a.shape[0] - h) // 2
    sx = (a.shape[1] - w) // 2
    dy = (shape[0] - h) // 2
    dx = (shape[1] - w) // 2
    out[dy:dy + h, dx:dx + w] = a[sy:sy + h, sx:sx + w]
    return out


def apply_shift(img, dy, dx):
    """Sub-pixel translate via FFT (bilinear-equivalent, no edge clamping)."""
    from scipy.ndimage import shift as ndshift
    return ndshift(img, (dy, dx), order=1, mode="reflect")
