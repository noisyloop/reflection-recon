"""
Reflection reconstruction from a video clip.

  FaceMesh ROI (eye / glasses)  ->  registered shift-and-add stack
  ->  Richardson-Lucy deconvolution with an estimated corneal PSF
  ->  contrast stretch.

MediaPipe and OpenCV are imported lazily so the PRNU tools work without them.
"""
import numpy as np
from skimage.restoration import richardson_lucy
from skimage.exposure import rescale_intensity

from .registration import register_rst, register_translation, apply_shift
from .psf import estimate_psf, gaussian_psf

# FaceMesh landmark indices (refined mesh, with iris)
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
         161, 246, 469, 470, 471, 472, 468]
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386,
         385, 384, 398, 474, 475, 476, 477, 473]


def _require(mod, name):
    if mod is None:
        raise ImportError(
            f"{name} is required for reflection extraction from video. "
            f"Install with: pip install {name}")


def roi_from_landmarks(lm, idx, W, H, pad=0.4):
    xs = [lm[i].x for i in idx]
    ys = [lm[i].y for i in idx]
    x0, x1 = min(xs) * W, max(xs) * W
    y0, y1 = min(ys) * H, max(ys) * H
    w, h = x1 - x0, y1 - y0
    x0 -= w * pad; x1 += w * pad
    y0 -= h * pad; y1 += h * pad
    return (max(0, int(x0)), max(0, int(y0)),
            min(W, int(x1)), min(H, int(y1)))


def extract_reflection(video_path, region="eye", pad=0.4, work=256,
                       max_frames=600, register="rst", deconv_iters=12,
                       psf_size=15, verbose=True):
    """Reconstruct the reflected content in an eye/glasses region of a clip.

    Returns a float grayscale image in [0, 1].
    """
    try:
        import cv2
    except ImportError:
        cv2 = None
    try:
        import mediapipe as mp
    except ImportError:
        mp = None
    _require(cv2, "opencv-python")
    _require(mp, "mediapipe")

    idx = R_EYE if region in ("eye", "reye") else (
        L_EYE if region == "leye" else R_EYE + L_EYE)

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    ref = None
    acc = None
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        x0, y0, x1, y1 = roi_from_landmarks(lm, idx, W, H, pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float64)
        crop = cv2.resize(crop, (work, work), interpolation=cv2.INTER_CUBIC) / 255.0

        if ref is None:
            ref = crop
            acc = crop.copy()
            n = 1
            continue

        # register current crop onto the reference, then add
        if register == "rst":
            t = register_rst(ref, crop)
            aligned = apply_shift(crop, *t["shift"])
        elif register == "translation":
            dy, dx = register_translation(ref, crop)
            aligned = apply_shift(crop, dy, dx)
        else:
            aligned = crop
        acc += aligned
        n += 1
        if verbose and n % 50 == 0:
            print(f"  stacked {n} frames")

    cap.release()
    if acc is None:
        raise RuntimeError("no face/eye detected in the clip")

    stacked = acc / n
    if verbose:
        print(f"  total frames stacked: {n}")

    # estimate PSF from the catchlight and deconvolve
    if deconv_iters > 0:
        psf = estimate_psf(stacked, size=psf_size)
        deconv = richardson_lucy(np.clip(stacked, 1e-6, 1),
                                 psf, num_iter=deconv_iters, clip=True)
    else:
        deconv = stacked

    lo, hi = np.percentile(deconv, [1, 99])
    return rescale_intensity(deconv, in_range=(lo, hi), out_range=(0.0, 1.0))
