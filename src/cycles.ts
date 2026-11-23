/**
 * PythonPine Cycles - Digital Signal Processing (DSP) & Cycle Indicators
 * Implementations of John Ehlers' DSP indicators.
 */

import { ArrayLike, ensureArray } from './utils.js';

export type HtSineResult = [Float64Array, Float64Array] & {
  sine: Float64Array;
  leadSine: Float64Array;
};

export type MamaFamaResult = [Float64Array, Float64Array] & {
  mama: Float64Array;
  fama: Float64Array;
};

/**
 * 1. Fisher Transform
 */
export function fisherTransform(
  source: ArrayLike,
  length = 10,
  smooth = true,
  smoothFactor = 0.33
): Float64Array {
  const src = ensureArray(source);
  const n = src.length;
  const fish = new Float64Array(n);
  let smoothedVal = 0.0;

  for (let i = 0; i < n; i++) {
    if (i < length) {
      fish[i] = 0.0;
      continue;
    }

    let highVal = -Infinity;
    let lowVal = Infinity;
    for (let j = Math.max(0, i - length + 1); j <= i; j++) {
      if (src[j] > highVal) highVal = src[j];
      if (src[j] < lowVal) lowVal = src[j];
    }

    const rangeVal = highVal !== lowVal ? highVal - lowVal : 1e-10;
    let normalized = (2 * (src[i] - lowVal)) / rangeVal - 1;
    normalized = Math.max(Math.min(normalized, 0.999), -0.999);

    let inputVal = normalized;
    if (smooth) {
      smoothedVal = smoothFactor * normalized + (1 - smoothFactor) * smoothedVal;
      inputVal = smoothedVal;
    }

    inputVal = Math.max(Math.min(inputVal, 0.999), -0.999);
    const fisher = 0.5 * Math.log((1 + inputVal) / (1 - inputVal));
    fish[i] = fisher;
  }

  return fish;
}

/**
 * 2. Hilbert Transform (Discrete FIR 4-coefficient filter)
 */
export function hilbertTransform(priceList: ArrayLike): Float64Array {
  const p = ensureArray(priceList);
  const n = p.length;
  const coeffs = [0.0962, 0.5769, 0.5769, 0.0962];
  const buffer = new Float64Array(n);

  for (let i = 6; i < n; i++) {
    buffer[i] =
      coeffs[0] * p[i] -
      coeffs[1] * p[i - 2] +
      coeffs[2] * p[i - 4] -
      coeffs[3] * p[i - 6];
  }

  return buffer;
}

/**
 * 3. Hilbert Sine and LeadSine
 */
export function htSine(priceList: ArrayLike): HtSineResult {
  const p = ensureArray(priceList);
  const q = hilbertTransform(p);
  const n = p.length;

  const sine = new Float64Array(n);
  const leadSine = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const inPhase = p[i];
    const quadrature = q[i];

    const phase = Math.atan2(quadrature, inPhase);
    sine[i] = Math.sin(phase);
    leadSine[i] = Math.sin(phase + Math.PI / 4); // 45° lead
  }

  const result = [sine, leadSine] as HtSineResult;
  result.sine = sine;
  result.leadSine = leadSine;
  return result;
}

/**
 * 4. Instantaneous Phase in degrees (0–360°)
 */
export function htPhase(priceList: ArrayLike): Float64Array {
  const p = ensureArray(priceList);
  const q = hilbertTransform(p);
  const n = p.length;
  const phaseDeg = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const inPhase = p[i];
    const quadrature = q[i];
    let deg = (Math.atan2(quadrature, inPhase) * 180) / Math.PI;
    if (deg < 0) {
      deg += 360;
    }
    phaseDeg[i] = deg;
  }

  return phaseDeg;
}

/**
 * 5. Laguerre RSI
 */
export function laguerreRsi(priceList: ArrayLike, gamma = 0.5): Float64Array {
  const p = ensureArray(priceList);
  const n = p.length;
  const rsiVals = new Float64Array(n);

  let l0 = 0.0;
  let l1 = 0.0;
  let l2 = 0.0;
  let l3 = 0.0;

  for (let i = 0; i < n; i++) {
    const price = p[i];
    const prevL0 = l0;
    const prevL1 = l1;
    const prevL2 = l2;
    const prevL3 = l3;

    l0 = (1 - gamma) * price + gamma * prevL0;
    l1 = -(1 - gamma) * l0 + prevL0 + gamma * prevL1;
    l2 = -(1 - gamma) * l1 + prevL1 + gamma * prevL2;
    l3 = -(1 - gamma) * l2 + prevL2 + gamma * prevL3;

    let cu = 0.0;
    let cd = 0.0;

    if (l0 >= l1) cu += l0 - l1;
    else cd += l1 - l0;

    if (l1 >= l2) cu += l1 - l2;
    else cd += l2 - l1;

    if (l2 >= l3) cu += l2 - l3;
    else cd += l3 - l2;

    const denom = cu + cd;
    rsiVals[i] = denom !== 0 ? cu / denom : 0.0;
  }

  return rsiVals;
}

/**
 * 6. MAMA / FAMA (MESA Adaptive Moving Average)
 */
export function mamaFama(
  priceList: ArrayLike,
  fastLimit = 0.5,
  slowLimit = 0.05
): MamaFamaResult {
  const p = ensureArray(priceList);
  const n = p.length;

  const mama = new Float64Array(n);
  const fama = new Float64Array(n);

  let phasePrev = 0.0;
  let mamaVal = 0.0;
  let famaVal = 0.0;

  for (let i = 0; i < n; i++) {
    if (i < 6) {
      mama[i] = 0.0;
      fama[i] = 0.0;
      continue;
    }

    const detrender =
      0.0962 * p[i] + 0.5769 * p[i - 2] - 0.5769 * p[i - 4] - 0.0962 * p[i - 6];
    let qPart = p[i - 3] - p[i - 6];
    if (Math.abs(qPart) < 1e-10) qPart = 1e-10;

    const phase = Math.atan(detrender / qPart);
    const deltaPhase = phase - phasePrev;
    phasePrev = phase;

    let alpha =
      Math.abs(deltaPhase) > 1e-5 ? fastLimit / Math.abs(deltaPhase) : slowLimit;
    alpha = Math.max(Math.min(alpha, fastLimit), slowLimit);

    mamaVal = alpha * p[i] + (1 - alpha) * mamaVal;
    famaVal = 0.5 * alpha * mamaVal + (1 - 0.5 * alpha) * famaVal;

    mama[i] = mamaVal;
    fama[i] = famaVal;
  }

  const result = [mama, fama] as MamaFamaResult;
  result.mama = mama;
  result.fama = fama;
  return result;
}

/**
 * 7. Super Smoother
 */
export function superSmoother(priceList: ArrayLike, period: number): Float64Array {
  const p = ensureArray(priceList);
  const n = p.length;
  const filt = new Float64Array(n);
  if (n === 0) return filt;

  const a1 = Math.exp((-1.414 * Math.PI) / period);
  const b1 = 2 * a1 * Math.cos((1.414 * Math.PI) / period);
  const c2 = b1;
  const c3 = -a1 * a1;
  const c1 = 1 - c2 - c3;

  for (let i = 0; i < n; i++) {
    if (i < 2) {
      filt[i] = p[i];
      continue;
    }
    filt[i] = (c1 * (p[i] + p[i - 1])) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2];
  }

  return filt;
}

/**
 * 8. Cyber Cycle
 */
export function cyberCycle(priceList: ArrayLike, alpha = 0.07): Float64Array {
  const p = ensureArray(priceList);
  const n = p.length;
  const cycle = new Float64Array(n);
  if (n === 0) return cycle;

  const c1 = Math.pow(1 - 0.5 * alpha, 2);
  const c2 = 2 * (1 - alpha);
  const c3 = Math.pow(1 - alpha, 2);

  for (let i = 0; i < n; i++) {
    if (i < 2) {
      cycle[i] = p[i];
      continue;
    }
    cycle[i] =
      c1 * (p[i] - 2 * p[i - 1] + p[i - 2]) +
      c2 * cycle[i - 1] -
      c3 * cycle[i - 2];
  }

  return cycle;
}
