"""
Command-line interface.

  recon prnu-enroll  <flat_frames_dir> -o cam.npy
  recon prnu-verify  <suspect_image>   -f cam.npy
  recon extract-reflection <clip.mp4> --region eye -o out.png
"""
import argparse
import glob
import os
import sys

import numpy as np


def _load_gray(path):
    from skimage.io import imread
    from skimage.color import rgb2gray
    img = imread(path)
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])
    return img.astype(np.float64) * (255.0 if img.max() <= 1.0 else 1.0)


def _crop_center(a, size):
    h, w = a.shape
    s = min(h, w, size)
    y = (h - s) // 2
    x = (w - s) // 2
    return a[y:y + s, x:x + s]


def cmd_enroll(args):
    from .prnu import estimate_fingerprint
    paths = sorted(sum([glob.glob(os.path.join(args.frames, e))
                        for e in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp")], []))
    if not paths:
        sys.exit(f"no images found in {args.frames}")
    print(f"enrolling from {len(paths)} frames ...")
    imgs = [_crop_center(_load_gray(p), args.size) for p in paths]
    K = estimate_fingerprint(imgs, sigma=args.sigma)
    np.save(args.output, K)
    print(f"fingerprint saved -> {args.output}   shape={K.shape}")


def cmd_verify(args):
    from .prnu import verify
    K = np.load(args.fingerprint)
    img = _crop_center(_load_gray(args.image), K.shape[0])
    r = verify(img, K, sigma=args.sigma)
    print(f"NCC : {r['ncc']:.5f}")
    print(f"PCE : {r['pce']:.2f}   (peak at {r['peak']})")
    verdict = ("MATCH (same sensor, likely)" if r["pce"] > 50
               else "inconclusive" if r["pce"] > 15 else "NO MATCH")
    print(f"==> {verdict}")


def cmd_extract(args):
    from .reflection import extract_reflection
    from skimage.io import imsave
    out = extract_reflection(
        args.clip, region=args.region, pad=args.pad, work=args.work,
        max_frames=args.max_frames, register=args.register,
        deconv_iters=args.iters, psf_size=args.psf_size)
    imsave(args.output, (np.clip(out, 0, 1) * 255).astype(np.uint8))
    print(f"reconstruction saved -> {args.output}")


def build_parser():
    p = argparse.ArgumentParser(prog="recon",
                                description="reflection recon + PRNU forensics")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("prnu-enroll", help="estimate a sensor fingerprint")
    e.add_argument("frames", help="directory of flat/bright frames")
    e.add_argument("-o", "--output", default="fingerprint.npy")
    e.add_argument("--sigma", type=float, default=3.0)
    e.add_argument("--size", type=int, default=512, help="center-crop size")
    e.set_defaults(func=cmd_enroll)

    v = sub.add_parser("prnu-verify", help="test an image against a fingerprint")
    v.add_argument("image")
    v.add_argument("-f", "--fingerprint", required=True)
    v.add_argument("--sigma", type=float, default=3.0)
    v.set_defaults(func=cmd_verify)

    x = sub.add_parser("extract-reflection", help="reconstruct reflected content")
    x.add_argument("clip")
    x.add_argument("-o", "--output", default="reflection.png")
    x.add_argument("--region", choices=["eye", "leye", "reye", "glasses"], default="eye")
    x.add_argument("--pad", type=float, default=0.4)
    x.add_argument("--work", type=int, default=256)
    x.add_argument("--max-frames", type=int, default=600)
    x.add_argument("--register", choices=["rst", "translation", "none"], default="rst")
    x.add_argument("--iters", type=int, default=12, help="Richardson-Lucy iterations")
    x.add_argument("--psf-size", type=int, default=15)
    x.set_defaults(func=cmd_extract)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
