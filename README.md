# recon-forensics

A research toolkit for two related imaging problems, in two forms:

- **`recon/`** — a Python offline toolkit (`numpy` / `scipy` / `scikit-image` / `PyWavelets` / `mediapipe`). The rigorous, reproducible bench.
- **`web/`** — a single self-contained HTML build with a hand-written, node-validated numerical core. The live, zero-install demo.

Both implement the same corrected methods:

1. **Wavelet denoiser** for the PRNU residual (Python: wavelet-MMSE / Mihcak; Web: Haar soft-threshold) — replaces the Gaussian high-pass of the earlier demo.
2. **FFT-based PCE** over the full 2-D circular cross-correlation surface — not a decoy-shift approximation.
3. **Log-polar (Fourier–Mellin) registration** for rotation + scale — not translation only.
4. **Estimated corneal PSF** (from the eye's catchlight) driving Richardson–Lucy deconvolution — not a fixed Gaussian.

Everything runs locally. No image data leaves the machine — no server, no upload.

> **Scope.** This exists to *understand* how reflection recovery and sensor fingerprinting work, and to demonstrate the privacy leakage they expose. It is deliberately honest about its limits ([below](#limitations)). It is a teaching / research instrument, not a courtroom pipeline.

---

## Repository layout

```
recon-forensics/
├── README.md
├── pyproject.toml            # console entry point: `recon`
├── requirements.txt
├── selftest.py               # synthetic ground-truth validation (Python)
├── recon/                    # Python package
│   ├── denoise.py            #   wavelet-MMSE denoiser -> PRNU residual
│   ├── prnu.py               #   fingerprint estimate, zero-mean, Wiener-DFT, NCC, FFT-PCE
│   ├── registration.py       #   phase-corr (translation) + Fourier-Mellin (rot/scale)
│   ├── psf.py                #   catchlight PSF estimation
│   ├── reflection.py         #   FaceMesh ROI -> registered stack -> RL deconv
│   └── cli.py                #   prnu-enroll / prnu-verify / extract-reflection
└── web/
    ├── reflection-recon.html #   standalone build (core is inlined)
    ├── recon-core.js         #   validated numerical core (source)
    ├── app.js                #   UI logic (source)
    ├── reflection-recon.template.html
    └── test-core.js          #   node validation of the JS core
```

---

## Python toolkit

### Install

```bash
pip install -r requirements.txt          # core: numpy scipy scikit-image PyWavelets
pip install opencv-python mediapipe      # only for extract-reflection from video
pip install -e .                         # optional: installs the `recon` command
```

### Validate

```bash
python selftest.py
# == PRNU identification ==        PASS  (PCE ~33000 match vs ~17 non-match)
# == Fourier-Mellin registration == PASS  (angle & scale recovered)
# == Richardson-Lucy deconvolution == PASS
```

### Use

```bash
# 1. estimate a sensor fingerprint from flat, bright frames
recon prnu-enroll ./flat_frames/ -o camA.npy

# 2. test whether an image came from that sensor
recon prnu-verify suspect.jpg -f camA.npy
#   NCC : 0.041
#   PCE : 312.7   (peak at (0, 0))
#   ==> MATCH (same sensor, likely)

# 3. reconstruct reflected content from an eye/glasses region of a clip
recon extract-reflection call.mp4 --region glasses --register rst --iters 12 -o out.png
```

Importable too:

```python
from recon import estimate_fingerprint, verify, register_rst, estimate_psf
K = estimate_fingerprint(list_of_flat_frames, sigma=3.0)
res = verify(test_image, K)          # -> {'ncc':..., 'pce':..., 'peak':(y,x)}
```

---

## Web build

Open `web/reflection-recon.html` over a **secure context** (`https://` or `localhost`) so the browser grants the camera:

```bash
cd web && python3 -m http.server 8000     # -> http://localhost:8000/reflection-recon.html
```

The file is standalone — the numerical core is inlined; only MediaPipe FaceMesh is pulled from a CDN. Two tabs: **reflection recon** (region select · shift-and-add with translation or rotation+scale registration · on-demand deconvolution · compare slider) and **sensor forensics** (enroll → verify with live NCC + FFT-PCE gauges).

### Rebuild after editing sources

`reflection-recon.html` is generated from the template + `recon-core.js` + `app.js`:

```bash
cd web
python3 - <<'PY'
tpl=open('reflection-recon.template.html').read()
out=tpl.replace('/*__RECON_CORE__*/',open('recon-core.js').read()).replace('/*__RECON_APP__*/',open('app.js').read())
open('reflection-recon.html','w').write(out)
PY
node test-core.js     # revalidate the core
```

---

## Methods

### PRNU (Lukáš–Fridrich–Goljan)

Every sensor imprints a fixed per-pixel gain pattern — **photo-response non-uniformity** — from silicon manufacturing variation. It's multiplicative and unique to a physical device.

1. **Residual** `W = I − F(I)`. Python uses a wavelet-MMSE denoiser (local variance across 3/5/7/9 windows, Wiener shrink); the web build uses a Haar soft-threshold. Both isolate the high-frequency, PRNU-bearing part.
2. **ML fingerprint** `K̂ = Σ(Wᵢ·Iᵢ) / Σ(Iᵢ²)` over many flat, bright frames — the weighting matters because PRNU scales with brightness.
3. **Clean** — zero-mean rows/columns, then Wiener-in-DFT (Python), to strip shared CFA/demosaic and periodic artifacts so cameras of the same *model* don't false-match.
4. **Detect** — correlate a test residual against the predicted signal `I·K̂`; report NCC and PCE.

### FFT-PCE (math note)

Peak-to-correlation-energy is computed on the full circular cross-correlation surface

```
C = IFFT( FFT(a) · conj(FFT(b)) )          # a = residual, b = predicted PRNU
```

normalized to unit energy. With the peak at `p`,

```
PCE = C(p)² / [ (1/(N−|Ω|)) · Σ_{s∉Ω} C(s)² ] · sign(C(p))
```

where `Ω` is a small exclusion neighborhood around the peak. Unlike raw NCC, PCE is robust to periodic artifacts and thresholds cleanly (**PCE > ~50** ≈ confident same-sensor match).

### Log-polar registration (math note)

Rotation and scale of an image become **translations** in the log-polar transform of its Fourier **magnitude** (which is itself shift-invariant) — the Fourier–Mellin trick (Reddy & Chatterji 1996). We window each image, take `|FFT|`, apply a high-pass emphasis to kill the DC blob, `warp_polar(..., scaling='log')`, then phase-correlate to read off `(angle, log-scale)`. The Fourier magnitude is symmetric, so rotation is only defined mod 180° with a sign ambiguity; we resolve it by testing the candidate transforms and keeping the one whose corrected image best aligns (highest NCC), then recover the residual translation by a final phase correlation.

### Estimated corneal PSF

The catchlight (Purkinje reflection) in an eye is the image of a near-point light source, so its captured shape is a direct sample of the system PSF at that location. We crop the brightest blob, background-subtract, normalize to sum 1, and feed it to Richardson–Lucy — a physically grounded PSF instead of an assumed Gaussian. Falls back to a defocus disc when no clean catchlight is present.

---

## Limitations

Stated plainly, because understanding the bounds is the point.

**Reflection recon**
- Recovers no detail the sensor never captured — not movie "enhance."
- At webcam resolution a corneal reflection spans ~20–40 px: light sources, window/monitor *shapes*, gross layout, and large high-contrast text — not fine screen content. Glasses (larger, flatter) shot on a tripod and integrated over hundreds of frames are where legible recovery happens.
- Registration corrects translation + rotation/scale, but not the space-varying corneal warp.
- Richardson–Lucy amplifies noise; run it after integration and stop before ringing.

**Sensor forensics**
- Operates on already-compressed 8-bit frames; RAW is far better.
- The web build uses a Haar denoiser and small tiles, so its non-match PCE floats higher than the Python wavelet-MMSE path — trust the NCC sign at low resolution. The Python toolkit is the reference implementation.

**Web build performance**
- A hand-written radix-2 FFT and log-polar search in JavaScript is far slower than the compiled Python libraries. Use `rot+scale` registration on short clips, run deconvolution on demand. For batch work, use the Python package.

---

## Where this technology is used in research

**Reflection / inverse rendering** — privacy attacks on video calls (Long et al., *Private Eye*, 2023; Backes et al., *Reflections of the Invisible*, IEEE S&P 2008–2010); corneal-glint eye tracking; eyes and chrome as light probes for graphics/AR (Nishino & Nayar, *The World in an Eye*, 2004); face-liveness / anti-spoofing; corneal topography.

**Sensor forensics** — source-camera identification and device linking; splice/tamper localization; deepfake detection via missing or inconsistent PRNU; device clustering for provenance in investigations; integrity checks for body/dash-cam evidence.

---

## Dual-use note

Both edges cut both ways. Reflection recovery is a privacy **attack** and the reason to defend (blur backgrounds, angle screens away, treat glasses on a call as a leak). PRNU is a forensic **defense** and a de-anonymization **vector** — the same fingerprint that ties a forgery to a device can link an anonymous source's photos across accounts. Hold both in mind before pointing either at real people.

---

## References

- J. Lukáš, J. Fridrich, M. Goljan — *Digital Camera Identification from Sensor Pattern Noise*, IEEE TIFS, 2006.
- M. Chen, J. Fridrich, M. Goljan, J. Lukáš — *Determining Image Origin and Integrity Using Sensor Noise*, IEEE TIFS, 2008.
- B. S. Reddy, B. N. Chatterji — *An FFT-Based Technique for Translation, Rotation and Scale-Invariant Image Registration*, IEEE TIP, 1996.
- M. Backes et al. — *Compromising Reflections, or How to Read LCD Monitors Around the Corner*, IEEE S&P, 2008.
- K. Nishino, S. K. Nayar — *The World in an Eye*, CVPR, 2004.
- W. H. Richardson (1972); L. B. Lucy (1974) — iterative deconvolution.

## License

TBD — suggest Apache 2.0 to match the rest of the noisyloop tooling.
