const C = require("./recon-core.js");
const W = 128, H = 128, N = W * H;

function rnd(seed) { let s = seed; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; }
const r = rnd(7);
function gauss() { return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r()); }

function smooth(a, W, H, rad) {                 // cheap box blur
  const o = new Float64Array(a.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0, c = 0;
    for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
      const yy = y + dy, xx = x + dx; if (yy < 0 || xx < 0 || yy >= H || xx >= W) continue; s += a[yy * W + xx]; c++;
    }
    o[y * W + x] = s / c;
  }
  return o;
}
function scene() { const a = new Float64Array(N); for (let i = 0; i < N; i++) a[i] = 0.5 + 0.15 * gauss(); return smooth(a, W, H, 6).map(v => Math.min(0.95, Math.max(0.15, v))); }
function textured() { const a = new Float64Array(N); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let v = 0.5 + 0.2 * Math.sin(2 * Math.PI * x / 11) + 0.15 * Math.sin(2 * Math.PI * y / 17); if (x > 30 && x < 70 && y > 40 && y < 80) v += 0.25; a[y * W + x] = v + 0.03 * gauss(); } return smooth(a, W, H, 1); }

function capture(sc, K) { const o = new Float64Array(N); for (let i = 0; i < N; i++) { let v = sc[i] * (1 + K[i]) + 0.01 * gauss(); v = Math.min(1, Math.max(0, v)); o[i] = Math.round(v * 255); } return o; }

console.log("== PRNU identification (JS core) ==");
const Ka = new Float64Array(N), Kb = new Float64Array(N);
for (let i = 0; i < N; i++) { Ka[i] = 0.03 * gauss(); Kb[i] = 0.03 * gauss(); }
let sa = null, sb = null;
for (let k = 0; k < 25; k++) { sa = C.fingerprintAccumulate(sa, capture(scene(), Ka), W, H, 1.5, 4); sb = C.fingerprintAccumulate(sb, capture(scene(), Kb), W, H, 1.5, 4); }
const Fa = C.fingerprintFinalize(sa), Fb = C.fingerprintFinalize(sb);
const test = capture(scene(), Ka);
const vAA = C.verify(test, Fa, W, H, 1.5, 4);
const vAB = C.verify(test, Fb, W, H, 1.5, 4);
console.log(`  test A vs fingerprint A : PCE=${vAA.pce.toFixed(1)}  NCC=${vAA.ncc.toFixed(4)}`);
console.log(`  test A vs fingerprint B : PCE=${vAB.pce.toFixed(1)}  NCC=${vAB.ncc.toFixed(4)}`);
const prnuOK = vAA.pce > 50 && vAB.pce < vAA.pce * 0.25;
console.log(`  -> ${prnuOK ? "PASS" : "FAIL"}\n`);

console.log("== log-polar registration (JS core) ==");
const img = textured();
const trueAngle = 6, trueScale = 1.06;
const moved = C.rotateScale(img, W, H, trueAngle, trueScale);
const reg = C.registerRST(img, moved, W, H);
// registerRST returns the correction (alignment) transform = inverse of applied warp
const expAngle = -trueAngle, expScale = 1 / trueScale;
console.log(`  applied     angle=+${trueAngle.toFixed(1)}  scale=${trueScale.toFixed(3)}`);
console.log(`  correction  angle=${reg.angle >= 0 ? "+" : ""}${reg.angle.toFixed(1)}  scale=${reg.scale.toFixed(3)}  (expect ${expAngle.toFixed(1)}, ${expScale.toFixed(3)})`);
const regOK = Math.abs(reg.angle - expAngle) < 2.5 && Math.abs(reg.scale - expScale) < 0.06;
console.log(`  -> ${regOK ? "PASS" : "FAIL"}\n`);

console.log("== estimated-PSF Richardson-Lucy (JS core) ==");
const tgt = textured();
const psf = C.gaussPSF(1.6);
function conv(img, psf) { const size = psf.size, r = size >> 1, o = new Float64Array(N); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let s = 0; for (let ky = 0; ky < size; ky++) for (let kx = 0; kx < size; kx++) { let iy = y + ky - r, ix = x + kx - r; iy = iy < 0 ? 0 : iy >= H ? H - 1 : iy; ix = ix < 0 ? 0 : ix >= W ? W - 1 : ix; s += img[iy * W + ix] * psf[ky * size + kx]; } o[y * W + x] = s; } return o; }
const b2 = conv(tgt, psf);
const clamped = new Float64Array(N); for (let i = 0; i < N; i++) clamped[i] = Math.min(1, Math.max(1e-6, b2[i]));
const rest = C.richardsonLucy(clamped, W, H, psf, 25);
function sharp(a) { let s = 0; for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) { const gx = a[y * W + x + 1] - a[y * W + x - 1], gy = a[(y + 1) * W + x] - a[(y - 1) * W + x]; s += gx * gx + gy * gy; } return s / N; }
const shBlur = sharp(b2), shRest = sharp(rest);
console.log(`  sharpness blurred=${shBlur.toExponential(2)} restored=${shRest.toExponential(2)}`);
const rlOK = shRest > shBlur * 1.5;
console.log(`  -> ${rlOK ? "PASS" : "FAIL"}\n`);

console.log("=".repeat(40));
console.log(`RESULT: ${[prnuOK, regOK, rlOK].filter(Boolean).length}/3 suites passed`);
