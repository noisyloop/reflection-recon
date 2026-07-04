/* recon-core.js — validated numerical core for the browser build.
   Implements the four corrected roadmap methods in vanilla JS:
     1. Haar wavelet soft-threshold denoiser (PRNU residual)
     2. FFT-based PCE over the full circular cross-correlation surface
     3. log-polar (Fourier-Mellin) rotation/scale registration
     4. estimated-PSF Richardson-Lucy deconvolution
   Works on power-of-two square tiles (default 256). */
(function (root) {
"use strict";

/* ----------------------------- 1-D radix-2 FFT ----------------------------- */
function fft(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr;
                 const ti = im[i]; im[i] = im[j]; im[j] = ti; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < (len >> 1); k++) {
        const a = i + k, b = a + (len >> 1);
        const vr = re[b] * cwr - im[b] * cwi;
        const vi = re[b] * cwi + im[b] * cwr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr;        im[a] += vi;
        const ncwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr; cwr = ncwr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

function fft2(re, im, W, H, inverse) {
  const rr = new Float64Array(W), ri = new Float64Array(W);
  for (let y = 0; y < H; y++) {
    const o = y * W;
    for (let x = 0; x < W; x++) { rr[x] = re[o + x]; ri[x] = im[o + x]; }
    fft(rr, ri, inverse);
    for (let x = 0; x < W; x++) { re[o + x] = rr[x]; im[o + x] = ri[x]; }
  }
  const cr = new Float64Array(H), ci = new Float64Array(H);
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) { cr[y] = re[y * W + x]; ci[y] = im[y * W + x]; }
    fft(cr, ci, inverse);
    for (let y = 0; y < H; y++) { re[y * W + x] = cr[y]; im[y * W + x] = ci[y]; }
  }
}

/* --------------------------------- helpers -------------------------------- */
function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
function zeroMean(a) { const m = mean(a); const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] - m; return o; }
function nccZero(a, b) {
  const A = zeroMean(a), B = zeroMean(b); let n = 0, da = 0, db = 0;
  for (let i = 0; i < A.length; i++) { n += A[i] * B[i]; da += A[i] * A[i]; db += B[i] * B[i]; }
  const d = Math.sqrt(da * db); return d < 1e-12 ? 0 : n / d;
}

/* -------------------- 2. FFT cross-correlation + PCE ---------------------- */
function crosscorr2(a, b, W, H) {
  const ar = zeroMean(a), ai = new Float64Array(W * H);
  const br = zeroMean(b), bi = new Float64Array(W * H);
  fft2(ar, ai, W, H, false); fft2(br, bi, W, H, false);
  const cr = new Float64Array(W * H), ci = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) {           // A * conj(B)
    cr[i] = ar[i] * br[i] + ai[i] * bi[i];
    ci[i] = ai[i] * br[i] - ar[i] * bi[i];
  }
  fft2(cr, ci, W, H, true);
  return cr;
}
function pce(a, b, W, H) {
  const C = crosscorr2(a, b, W, H), n = W * H;
  let e = 0; for (let i = 0; i < n; i++) e += C[i] * C[i];
  const norm = Math.sqrt(e / n) + 1e-30;
  let pk = 0, pi = 0;
  for (let i = 0; i < n; i++) { const v = C[i] / norm; C[i] = v; if (Math.abs(v) > Math.abs(pk)) { pk = v; pi = i; } }
  const py = (pi / W) | 0, px = pi % W;
  let energy = 0, cnt = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const dy = Math.min((y - py + H) % H, (py - y + H) % H);
    const dx = Math.min((x - px + W) % W, (px - x + W) % W);
    if (dy <= 1 && dx <= 1) continue;
    energy += C[y * W + x] * C[y * W + x]; cnt++;
  }
  energy /= cnt;
  return { pce: energy < 1e-30 ? 0 : pk * pk / energy * Math.sign(pk),
           peak: [py, px], ncc: nccZero(a, b) };
}

/* --------------------- 1. Haar soft-threshold denoiser -------------------- */
const S2 = Math.SQRT1_2;
function haarFwd(a, W, H, w, h) {                 // one level on top-left w x h
  const t = new Float64Array(w);
  for (let y = 0; y < h; y++) {                   // rows
    const o = y * W, half = w >> 1;
    for (let i = 0; i < half; i++) { t[i] = (a[o + 2 * i] + a[o + 2 * i + 1]) * S2; t[half + i] = (a[o + 2 * i] - a[o + 2 * i + 1]) * S2; }
    for (let i = 0; i < w; i++) a[o + i] = t[i];
  }
  const tc = new Float64Array(h);
  for (let x = 0; x < w; x++) {                   // cols
    const half = h >> 1;
    for (let i = 0; i < half; i++) { tc[i] = (a[2 * i * W + x] + a[(2 * i + 1) * W + x]) * S2; tc[half + i] = (a[2 * i * W + x] - a[(2 * i + 1) * W + x]) * S2; }
    for (let i = 0; i < h; i++) a[i * W + x] = tc[i];
  }
}
function haarInv(a, W, H, w, h) {
  const tc = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    const half = h >> 1;
    for (let i = 0; i < half; i++) { const s = a[i * W + x], d = a[(half + i) * W + x]; tc[2 * i] = (s + d) * S2; tc[2 * i + 1] = (s - d) * S2; }
    for (let i = 0; i < h; i++) a[i * W + x] = tc[i];
  }
  const t = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const o = y * W, half = w >> 1;
    for (let i = 0; i < half; i++) { const s = a[o + i], d = a[o + half + i]; t[2 * i] = (s + d) * S2; t[2 * i + 1] = (s - d) * S2; }
    for (let i = 0; i < w; i++) a[o + i] = t[i];
  }
}
function softThresholdDetails(a, W, H, w, h, thr) { // shrink LH,HL,HH of this level
  const hw = w >> 1, hh = h >> 1;
  const soft = v => { const s = Math.sign(v), m = Math.abs(v) - thr; return m > 0 ? s * m : 0; };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (x < hw && y < hh) continue;                // LL — keep
    a[y * W + x] = soft(a[y * W + x]);
  }
}
function waveletDenoise(img, W, H, sigma, levels) {
  const a = Float64Array.from(img);
  let w = W, h = H;
  const sizes = [];
  for (let l = 0; l < levels && w > 1 && h > 1; l++) { haarFwd(a, W, H, w, h); sizes.push([w, h]); w >>= 1; h >>= 1; }
  // universal-ish threshold from noise sigma
  const thr = sigma * Math.sqrt(2 * Math.log(W * H));
  for (let l = sizes.length - 1; l >= 0; l--) { const [cw, ch] = sizes[l]; softThresholdDetails(a, W, H, cw, ch, thr); }
  for (let l = sizes.length - 1; l >= 0; l--) { const [cw, ch] = sizes[l]; haarInv(a, W, H, cw, ch); }
  return a;
}
function noiseResidual(img, W, H, sigma, levels) {
  const den = waveletDenoise(img, W, H, sigma || 3, levels || 4);
  const r = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) r[i] = img[i] - den[i];
  return r;
}

/* ----------------------------- PRNU fingerprint --------------------------- */
function zeroMeanRC(K, W, H) {
  for (let y = 0; y < H; y++) { let m = 0; for (let x = 0; x < W; x++) m += K[y * W + x]; m /= W; for (let x = 0; x < W; x++) K[y * W + x] -= m; }
  for (let x = 0; x < W; x++) { let m = 0; for (let y = 0; y < H; y++) m += K[y * W + x]; m /= H; for (let y = 0; y < H; y++) K[y * W + x] -= m; }
  return K;
}
function fingerprintAccumulate(state, img, W, H, sigma, levels) {
  if (!state) state = { num: new Float64Array(W * H), den: new Float64Array(W * H), n: 0, W, H };
  const w = noiseResidual(img, W, H, sigma, levels);
  for (let i = 0; i < W * H; i++) { state.num[i] += w[i] * img[i]; state.den[i] += img[i] * img[i]; }
  state.n++;
  return state;
}
function fingerprintFinalize(state) {
  const { num, den, W, H } = state, K = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) K[i] = num[i] / (den[i] + 1);
  return zeroMeanRC(K, W, H);
}
function verify(img, K, W, H, sigma, levels) {
  const w = noiseResidual(img, W, H, sigma, levels);
  const pred = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) pred[i] = img[i] * K[i];
  return pce(w, pred, W, H);
}

/* --------------- 3. log-polar (Fourier-Mellin) registration --------------- */
function fftMag(img, W, H) {
  const re = Float64Array.from(img), im = new Float64Array(W * H);
  // Hann window + RC high-pass emphasis applied after shift
  for (let y = 0; y < H; y++) { const wy = 0.5 - 0.5 * Math.cos(2 * Math.PI * y / (H - 1)); for (let x = 0; x < W; x++) { const wx = 0.5 - 0.5 * Math.cos(2 * Math.PI * x / (W - 1)); re[y * W + x] *= wy * wx; } }
  fft2(re, im, W, H, false);
  const mag = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {   // fftshift + |.|
    const sy = (y + (H >> 1)) % H, sx = (x + (W >> 1)) % W;
    const i = sy * W + sx;
    let m = Math.hypot(re[i], im[i]);
    const u = y / (H - 1) - 0.5, v = x / (W - 1) - 0.5;         // RC high-pass
    const X = Math.cos(Math.PI * u) * Math.cos(Math.PI * v);
    mag[y * W + x] = m * (1 - X) * (2 - X);
  }
  return mag;
}
function logPolar(mag, W, H, nAng, nRad) {
  const cx = W / 2, cy = H / 2, maxR = Math.min(W, H) / 2;
  const klog = Math.log(maxR) / nRad;
  const out = new Float64Array(nAng * nRad);
  for (let a = 0; a < nAng; a++) {
    const th = Math.PI * a / nAng;               // 0..pi (mag symmetric)
    const ct = Math.cos(th), st = Math.sin(th);
    for (let r = 0; r < nRad; r++) {
      const rad = Math.exp(r * klog);
      const fx = cx + rad * ct, fy = cy + rad * st;
      const x0 = Math.floor(fx), y0 = Math.floor(fy);
      if (x0 < 0 || y0 < 0 || x0 >= W - 1 || y0 >= H - 1) continue;
      const dx = fx - x0, dy = fy - y0;
      const v = mag[y0 * W + x0] * (1 - dx) * (1 - dy) + mag[y0 * W + x0 + 1] * dx * (1 - dy)
              + mag[(y0 + 1) * W + x0] * (1 - dx) * dy + mag[(y0 + 1) * W + x0 + 1] * dx * dy;
      out[a * nRad + r] = v;
    }
  }
  return { lp: out, klog, maxR };
}
function phaseCorr(a, b, W, H) {                 // returns {dy,dx} integer + parabolic
  const ar = Float64Array.from(a), ai = new Float64Array(W * H);
  const br = Float64Array.from(b), bi = new Float64Array(W * H);
  fft2(ar, ai, W, H, false); fft2(br, bi, W, H, false);
  const cr = new Float64Array(W * H), ci = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const rr = ar[i] * br[i] + ai[i] * bi[i], ii = ai[i] * br[i] - ar[i] * bi[i];
    const mag = Math.hypot(rr, ii) + 1e-12;
    cr[i] = rr / mag; ci[i] = ii / mag;          // cross-power spectrum (phase only)
  }
  fft2(cr, ci, W, H, true);
  let pk = -1e30, pi = 0;
  for (let i = 0; i < W * H; i++) if (cr[i] > pk) { pk = cr[i]; pi = i; }
  let py = (pi / W) | 0, px = pi % W;
  const at = (y, x) => cr[((y + H) % H) * W + ((x + W) % W)];
  const spx = parab(at(py, px - 1), pk, at(py, px + 1));
  const spy = parab(at(py - 1, px), pk, at(py + 1, px));
  if (py > H / 2) py -= H; if (px > W / 2) px -= W;
  return { dy: py + spy, dx: px + spx, peak: pk };
}
function parab(l, c, r) { const d = l - 2 * c + r; return Math.abs(d) < 1e-9 ? 0 : 0.5 * (l - r) / d; }

function registerRST(ref, mov, W, H) {
  const nAng = 256, nRad = W;                     // both power-of-two for the FFT
  const lpR = logPolar(fftMag(ref, W, H), W, H, nAng, nRad);
  const lpM = logPolar(fftMag(mov, W, H), W, H, nAng, nRad);
  const pc = phaseCorr(lpR.lp, lpM.lp, nRad, nAng);   // array: nAng rows x nRad cols
  const angMag = (180 / nAng) * pc.dy;            // degrees (magnitude, mod 180)
  const scaleMag = Math.exp(pc.dx * lpR.klog);
  // resolve rotation sign / 180-deg and scale-direction ambiguity by best NCC
  let best = null;
  const angles = [angMag, -angMag, angMag + 180, -angMag + 180];
  const scales = [scaleMag, 1 / scaleMag];
  for (const a of angles) for (const sc of scales) {
    const corr = rotateScale(mov, W, H, a, sc);
    const t = phaseCorr(ref, corr, W, H);
    const aligned = shiftImg(corr, W, H, t.dy, t.dx);
    const s = nccZero(ref, aligned);
    if (!best || s > best.s) best = { s, angle: a, scale: sc, dy: t.dy, dx: t.dx };
  }
  const angle = ((best.angle + 180) % 360) - 180;
  return { angle, scale: best.scale, dy: best.dy, dx: best.dx, score: best.s };
}
function rotateScale(img, W, H, deg, scale) {     // bilinear, about center
  const out = new Float64Array(W * H), cx = W / 2, cy = H / 2;
  const rad = -deg * Math.PI / 180, c = Math.cos(rad) / scale, s = Math.sin(rad) / scale;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const xr = x - cx, yr = y - cy;
    const sx = cx + (c * xr - s * yr), sy = cy + (s * xr + c * yr);
    const x0 = Math.floor(sx), y0 = Math.floor(sy);
    if (x0 < 0 || y0 < 0 || x0 >= W - 1 || y0 >= H - 1) continue;
    const dx = sx - x0, dy = sy - y0;
    out[y * W + x] = img[y0 * W + x0] * (1 - dx) * (1 - dy) + img[y0 * W + x0 + 1] * dx * (1 - dy)
                   + img[(y0 + 1) * W + x0] * (1 - dx) * dy + img[(y0 + 1) * W + x0 + 1] * dx * dy;
  }
  return out;
}
function shiftImg(img, W, H, dy, dx) {
  const out = new Float64Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const sy = y - dy, sx = x - dx;
    const y0 = Math.floor(sy), x0 = Math.floor(sx);
    let yy0 = y0 < 0 ? 0 : y0 >= H - 1 ? H - 2 : y0;
    let xx0 = x0 < 0 ? 0 : x0 >= W - 1 ? W - 2 : x0;
    const fy = sy - yy0, fx = sx - xx0;
    out[y * W + x] = img[yy0 * W + xx0] * (1 - fx) * (1 - fy) + img[yy0 * W + xx0 + 1] * fx * (1 - fy)
                   + img[(yy0 + 1) * W + xx0] * (1 - fx) * fy + img[(yy0 + 1) * W + xx0 + 1] * fx * fy;
  }
  return out;
}

/* ------------- 4. estimated-PSF Richardson-Lucy deconvolution ------------- */
function conv2(img, W, H, psf, pw, ph) {
  const out = new Float64Array(W * H), rx = pw >> 1, ry = ph >> 1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0;
    for (let ky = 0; ky < ph; ky++) for (let kx = 0; kx < pw; kx++) {
      let iy = y + ky - ry, ix = x + kx - rx;
      iy = iy < 0 ? 0 : iy >= H ? H - 1 : iy; ix = ix < 0 ? 0 : ix >= W ? W - 1 : ix;
      s += img[iy * W + ix] * psf[ky * pw + kx];
    }
    out[y * W + x] = s;
  }
  return out;
}
function estimatePSF(eye, W, H, size) {           // from brightest catchlight blob
  if ((size & 1) === 0) size++;
  const r = size >> 1;
  let pk = -1e30, pi = 0;
  for (let i = 0; i < W * H; i++) if (eye[i] > pk) { pk = eye[i]; pi = i; }
  let y0 = Math.min(Math.max((pi / W) | 0, r), H - r - 1);
  let x0 = Math.min(Math.max(pi % W, r), W - r - 1);
  const patch = new Float64Array(size * size); let bg = 1e30;
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) { const v = eye[(y0 + y) * W + (x0 + x)]; if (v < bg) bg = v; }
  let s = 0;
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) { const v = Math.max(0, eye[(y0 + y) * W + (x0 + x)] - bg); patch[(y + r) * size + (x + r)] = v; s += v; }
  if (s < 1e-6) { const g = gaussPSF(Math.max(1, size / 6)); return g; }
  for (let i = 0; i < patch.length; i++) patch[i] /= s;
  patch.size = size; return patch;
}
function gaussPSF(sigma) {
  const r = Math.max(1, Math.round(sigma * 3)), size = 2 * r + 1, k = new Float64Array(size * size); let s = 0;
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) { const v = Math.exp(-(x * x + y * y) / (2 * sigma * sigma)); k[(y + r) * size + (x + r)] = v; s += v; }
  for (let i = 0; i < k.length; i++) k[i] /= s; k.size = size; return k;
}
function richardsonLucy(obs, W, H, psf, iters) {   // obs in [0,1]
  const size = psf.size, flip = new Float64Array(psf.length);
  for (let i = 0; i < psf.length; i++) flip[i] = psf[psf.length - 1 - i];
  let u = Float64Array.from(obs);
  for (let t = 0; t < iters; t++) {
    const cu = conv2(u, W, H, psf, size, size);
    const ratio = new Float64Array(W * H);
    for (let i = 0; i < W * H; i++) ratio[i] = obs[i] / (cu[i] + 1e-4);
    const corr = conv2(ratio, W, H, flip, size, size);
    for (let i = 0; i < W * H; i++) { u[i] *= corr[i]; if (u[i] > 1) u[i] = 1; else if (u[i] < 0) u[i] = 0; }
  }
  return u;
}

const API = { fft, fft2, nccZero, crosscorr2, pce, waveletDenoise, noiseResidual,
  zeroMeanRC, fingerprintAccumulate, fingerprintFinalize, verify,
  registerRST, rotateScale, shiftImg, phaseCorr, estimatePSF, gaussPSF, richardsonLucy };
if (typeof module !== "undefined" && module.exports) module.exports = API;
else root.ReconCore = API;
})(typeof self !== "undefined" ? self : this);
