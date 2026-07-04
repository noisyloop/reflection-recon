# reflection-recon

A client-side forensic imaging bench for two related research problems:

1. **Reflection reconstruction** — isolating and enhancing what's reflected in a subject's eyes, glasses, or any specular surface in a camera feed.
2. **Sensor forensics (PRNU)** — estimating a camera's photo-response non-uniformity fingerprint and testing whether an image came from it.

Everything runs in the browser. No frame ever leaves the tab — no server, no upload, no network round-trip for the imagery.

> **Scope.** This is a teaching and research instrument for understanding how reflection recovery and sensor fingerprinting work, and for demonstrating the privacy leakage they expose. It is deliberately honest about its limits (see [Limitations](#limitations)). It is not a courtroom pipeline and does not claim to be.

---

## Running it

Camera access requires a secure context, so opening the file directly (`file://`) will not grant the webcam. Serve it over `https://` or `localhost`:

```bash
# any static server works
python3 -m http.server 8000
# then open http://localhost:8000/reflection-recon.html
```

Or deploy the single file to any static host (Vercel, Cloudflare Pages, Netlify). No build step — it's one self-contained HTML file that pulls MediaPipe from a CDN at runtime.

**Requirements:** a modern browser (Chromium/Firefox/Safari) with WebGL and `getUserMedia`. Face/eye tracking uses MediaPipe FaceMesh loaded from jsDelivr; the manual-ROI and full-frame modes work without it.

---

## Mode 1 — Reflection recon

The cornea is a convex mirror (~7.8 mm radius); eyeglass lenses and any shiny object are larger, flatter mirrors. Whatever the subject faces is reflected back toward the camera. This mode isolates that reflective region and pushes it through a reconstruction pipeline.

### Pipeline

```
source ─▶ FaceMesh landmark lock ─▶ ROI crop (native res)
       ─▶ sub-pixel registration ─▶ temporal integration
       ─▶ luma ─▶ Richardson–Lucy deconv ─▶ auto-stretch
       ─▶ brightness/contrast/gamma ─▶ unsharp ─▶ raw⇄processed blend
```

### Controls

| Group | What it does |
|---|---|
| **Target region** | Right eye · left eye · both · glasses (widened lens box) · manual drag-ROI · full frame |
| **Integration** | Temporal averaging (denoise) or max-hold, with **sub-pixel registration** to align frames before stacking |
| **Deconvolution** | Richardson–Lucy iterative deblur with an adjustable Gaussian (defocus) PSF |
| **Enhance** | Luma extraction, 1/99-percentile auto-stretch, brightness/contrast/gamma, unsharp mask |
| **Compare** | Wipe between the raw single-frame ROI and the fully processed result |

### How to get a usable result

Reflection recovery rewards a static, high-signal setup:

- Mount the camera on a **tripod** — registration only corrects translation, not the corneal warp.
- Use a **bright, high-contrast** target (a monitor showing large text, a window).
- **Glasses beat eyes** — larger, flatter reflective surface, far more legible.
- Turn on **integration + registration** and let it stack hundreds of frames.
- Add **Richardson–Lucy** last, a few iterations at a time. Stop before ringing appears.
- Use the **compare slider** to confirm you recovered real structure rather than amplifying noise.

---

## Mode 2 — Sensor forensics (PRNU)

Every image sensor imprints a fixed, per-pixel gain pattern from silicon manufacturing variation — **photo-response non-uniformity**. It's multiplicative, survives compression to a degree, and is unique to a physical device. It's a fingerprint.

This implements the canonical Lukáš–Fridrich–Goljan (2006) workflow:

1. **Residual** — for each frame `I`, denoise and take `W = I − F(I)`.
2. **ML fingerprint** — because PRNU scales with brightness, estimate `K̂ = Σ(Wᵢ·Iᵢ) / Σ(Iᵢ²)` over many *flat, bright, smooth* frames.
3. **Clean** — zero-mean every row and column to strip shared CFA/demosaic artifacts, so cameras of the same *model* don't false-match.
4. **Detect** — correlate a test frame's residual against the predicted signal `I·K̂`; report **NCC** and a **PCE** (peak-to-correlation-energy) statistic.

### Workflow in the tool

1. Point the camera at a **blank bright wall** (or defocus toward a bright sky). Click **enroll fingerprint** and let it accumulate frames.
2. Switch **fingerprint view** to `K̂` to see the estimated pattern.
3. Click **run verify**. Point at a different scene from the *same* camera → NCC/PCE should lift. Load an image from a *different* device → it should collapse to no-match.

Rule of thumb: **PCE > ~50** is a confident same-sensor match in the literature. Webcam JPEG and low resolution depress the numbers, so treat this as a demonstration of the method, not evidence.

---

## Limitations

Stated plainly, because the whole point is to understand the real bounds:

**Reflection recon**
- Recovers *no* detail the sensor never captured — it is not movie "enhance."
- At webcam resolution a corneal reflection spans ~20–40 px: you get light sources, window/monitor *shapes*, gross layout, and large high-contrast text — not fine screen content.
- Registration corrects translation only (no rotation, scale, or the space-varying corneal warp).
- Richardson–Lucy assumes a spatially invariant Gaussian PSF, which the cornea is not — it's an approximation, and it amplifies noise.
- Motion, autofocus hunting, and webcam compression are the real enemies.

**Sensor forensics**
- Operates on already-compressed 8-bit frames; RAW is far better.
- Uses a Gaussian high-pass as a fast stand-in for the wavelet denoiser used in production PRNU.
- Computes an approximate PCE over decoy shifts rather than a full FFT correlation surface.

---

## Where this technology is used in research

**Reflection / inverse rendering**
- Privacy attacks on video calls — recovering on-screen text from eyeglass reflections (*Private Eye*, Long et al. 2023) and screen/room reconstruction from the eye (Backes et al., *Reflections of the Invisible*, IEEE S&P 2008–2010).
- Eye tracking via corneal-glint (Purkinje) reflections — the basis of most commercial eye trackers.
- Environment capture for graphics/AR, using eyes and chrome surfaces as light probes (Nishino & Nayar, *The World in an Eye*, 2004).
- Face-liveness / anti-spoofing, and corneal topography in ophthalmology.

**Sensor forensics**
- Source-camera identification and device linking.
- Tamper/splice localization (a forged region lacks the fingerprint).
- Deepfake / synthetic-media detection via missing or inconsistent PRNU.
- Device clustering for image provenance in investigations.
- Integrity verification for body-cam / dash-cam evidence.

---

## Dual-use note

Both techniques cut both ways.

Reflection recovery is simultaneously a privacy **attack** and the reason to study it **defensively** — blur backgrounds, angle screens away from the camera, and treat glasses on a call as a potential leak.

PRNU is a forensic **defense** that is also a de-anonymization **vector** — the same fingerprint that ties a forgery to a device can link an anonymous source's photos across accounts. Hold both edges in mind before pointing either at real people.

---

## Tech

- Single self-contained HTML file, vanilla JS, 2D canvas.
- [MediaPipe FaceMesh](https://developers.google.com/mediapipe) (refined landmarks, iris) for eye/glasses ROI.
- All DSP (registration, Gaussian/separable convolution, Richardson–Lucy, PRNU residual/correlation) implemented in-file with typed arrays.

## Roadmap

- Wavelet (Haar/Daubechies) denoiser to replace the Gaussian residual in PRNU.
- True FFT-based PCE over the full correlation surface.
- Phase-correlation registration for rotation/scale, not just translation.
- Estimated corneal PSF for physically grounded deconvolution.

## License

MIT
