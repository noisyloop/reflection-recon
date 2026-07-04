"""
Synthetic ground-truth validation of the DSP core (no camera / mediapipe needed).

  * PRNU: two virtual sensors with distinct fingerprints. Enroll each, then
    verify held-out frames -> matching camera must give high PCE, the other low.
  * Registration: apply a known rotation+scale+shift, recover it.
  * Deconvolution: blur a target with a known PSF, deconvolve, check it sharpens.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from recon.prnu import estimate_fingerprint, verify
from recon.registration import register_rst
from recon.psf import gaussian_psf
from skimage.restoration import richardson_lucy
from skimage.transform import rotate, rescale
from scipy.ndimage import shift as ndshift

rng = np.random.default_rng(0)
H = W = 256


def scene():
    """A smooth, bright, flat-ish scene (good for PRNU) with mild content."""
    base = gaussian_filter(rng.normal(0.5, 0.15, (H, W)), 8)
    return np.clip(base, 0.15, 0.95)


def textured_scene():
    """A scene with real structure (edges/gratings) for registration tests."""
    y, x = np.mgrid[0:H, 0:W].astype(float)
    img = 0.5 + 0.2 * np.sin(2 * np.pi * x / 23) + 0.15 * np.sin(2 * np.pi * y / 31)
    img[60:120, 80:170] += 0.25          # a block
    img += 0.05 * rng.standard_normal((H, W))
    return np.clip(gaussian_filter(img, 0.6), 0, 1)


def capture(scene_img, K, read_noise=0.01):
    """Sensor model: I = scene * (1 + K) + read noise, quantized to 8-bit."""
    img = scene_img * (1.0 + K) + rng.normal(0, read_noise, (H, W))
    img = np.clip(img, 0, 1)
    return np.round(img * 255) / 255.0 * 255.0  # emulate 8-bit


def prnu_test():
    print("== PRNU identification ==")
    K_a = rng.normal(0, 0.03, (H, W))   # sensor A fingerprint
    K_b = rng.normal(0, 0.03, (H, W))   # sensor B fingerprint

    enroll_a = [capture(scene(), K_a) for _ in range(25)]
    enroll_b = [capture(scene(), K_b) for _ in range(25)]
    Fa = estimate_fingerprint(enroll_a, sigma=1.5)
    Fb = estimate_fingerprint(enroll_b, sigma=1.5)

    # held-out frames from sensor A
    test_a = capture(scene(), K_a)
    ra_a = verify(test_a, Fa, sigma=1.5)     # A vs A  -> should MATCH
    ra_b = verify(test_a, Fb, sigma=1.5)     # A vs B  -> should NOT match
    print(f"  test A vs fingerprint A : PCE={ra_a['pce']:8.1f}  NCC={ra_a['ncc']:+.4f}")
    print(f"  test A vs fingerprint B : PCE={ra_b['pce']:8.1f}  NCC={ra_b['ncc']:+.4f}")
    ok = ra_a["pce"] > 50 and ra_b["pce"] < ra_a["pce"] * 0.2
    print(f"  -> {'PASS' if ok else 'FAIL'}: correct camera separates cleanly\n")
    return ok


def registration_test():
    print("== Fourier-Mellin registration ==")
    img = textured_scene()
    true_angle, true_scale, true_shift = 7.0, 1.08, (4.0, -6.0)
    moved = rotate(img, true_angle, order=1, preserve_range=True)
    moved = rescale(moved, true_scale, order=1, preserve_range=True, anti_aliasing=True)
    # center-crop back to HxW
    y = (moved.shape[0] - H) // 2
    x = (moved.shape[1] - W) // 2
    moved = moved[y:y + H, x:x + W]
    moved = ndshift(moved, true_shift, order=1, mode="reflect")

    r = register_rst(img, moved)
    print(f"  true  angle={true_angle:+.2f}  scale={true_scale:.3f}")
    print(f"  est   angle={r['angle']:+.2f}  scale={r['scale']:.3f}  shift={tuple(round(v,1) for v in r['shift'])}")
    ok = abs(r["angle"] - true_angle) < 2.0 and abs(r["scale"] - true_scale) < 0.06
    print(f"  -> {'PASS' if ok else 'FAIL'}: rotation & scale recovered\n")
    return ok


def deconv_test():
    print("== Richardson-Lucy deconvolution ==")
    target = scene()
    psf = gaussian_psf(2.0)
    blurred = gaussian_filter(target, 2.0)
    restored = richardson_lucy(np.clip(blurred, 1e-6, 1), psf, num_iter=25, clip=True)

    def sharp(a):  # gradient energy as a sharpness proxy
        gy, gx = np.gradient(a)
        return float(np.mean(gx**2 + gy**2))
    s_blur, s_rest = sharp(blurred), sharp(restored)
    print(f"  sharpness  blurred={s_blur:.2e}  restored={s_rest:.2e}")
    ok = s_rest > s_blur * 1.5
    print(f"  -> {'PASS' if ok else 'FAIL'}: deconvolution recovers high-frequency detail\n")
    return ok


if __name__ == "__main__":
    results = [prnu_test(), registration_test(), deconv_test()]
    print("=" * 44)
    print(f"RESULT: {sum(results)}/{len(results)} suites passed")
