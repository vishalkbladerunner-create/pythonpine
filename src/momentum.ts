/**
 * PythonPine Momentum - High-Performance Vectorized Indicators
 */

import {
  ArrayLike,
  ensureArray,
  rollingMax,
  rollingMin,
  rollingMean,
  rollingSum,
  ewmMean,
  wilderSmooth,
  padNan,
  diff,
  shift,
  trueRange
} from './utils.js';

export type MacdResult = [Float64Array, Float64Array, Float64Array] & {
  macd: Float64Array;
  signal: Float64Array;
  histogram: Float64Array;
};

export type StochasticResult = [Float64Array, Float64Array] & {
  k: Float64Array;
  d: Float64Array;
};

export type AdxResult = [Float64Array, Float64Array, Float64Array] & {
  adx: Float64Array;
  plusDi: Float64Array;
  minusDi: Float64Array;
};

export type StochRsiResult = [Float64Array, Float64Array] & {
  k: Float64Array;
  d: Float64Array;
};

/**
 * Relative Strength Index (RSI) - Vectorized.
 * Uses Wilder's smoothing (RMA) for Pine Script parity.
 * Warmup: length bars
 */
export function rsi(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const n = close.length;
  if (n === 0) return new Float64Array(0);

  const delta = diff(close, 1);
  const gains = new Float64Array(n);
  const losses = new Float64Array(n);

  for (let i = 1; i < n; i++) {
    const d = delta[i];
    if (d > 0) gains[i] = d;
    else if (d < 0) losses[i] = -d;
  }

  const avgGain = wilderSmooth(gains, length, 1);
  const avgLoss = wilderSmooth(losses, length, 1);
  const rsiValues = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const al = avgLoss[i] === 0 ? 1e-10 : avgLoss[i];
    const rs = avgGain[i] / al;
    rsiValues[i] = 100 - (100 / (1 + rs));
  }

  return padNan(rsiValues, length);
}

/**
 * MACD (Moving Average Convergence Divergence) - Vectorized.
 * Returns: [macdLine, signalLine, histogram] & { macd, signal, histogram }
 * Warmup: slow + signal bars
 */
export function macd(
  source: ArrayLike,
  fast = 12,
  slow = 26,
  signal = 9
): MacdResult {
  const close = ensureArray(source);
  const fastEma = ewmMean(close, fast);
  const slowEma = ewmMean(close, slow);
  const n = close.length;

  const macdLine = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    macdLine[i] = fastEma[i] - slowEma[i];
  }

  const signalLine = ewmMean(macdLine, signal);
  const histogram = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    histogram[i] = macdLine[i] - signalLine[i];
  }

  const warmup = slow + signal;
  padNan(macdLine, slow);
  padNan(signalLine, warmup);
  padNan(histogram, warmup);

  const result = [macdLine, signalLine, histogram] as MacdResult;
  result.macd = macdLine;
  result.signal = signalLine;
  result.histogram = histogram;
  return result;
}

/**
 * Stochastic Oscillator - Vectorized.
 * Returns: [k, d] & { k, d }
 */
export function stochastic(
  close: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  length = 14,
  smoothK = 3,
  smoothD = 3
): StochasticResult {
  const c = ensureArray(close);
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = c.length;

  const hh = rollingMax(h, length);
  const ll = rollingMin(l, length);
  const rawK = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const denom = hh[i] - ll[i] === 0 ? 1e-10 : hh[i] - ll[i];
    rawK[i] = (100 * (c[i] - ll[i])) / denom;
  }

  const k = padNan(rollingMean(rawK, smoothK), length);
  const d = padNan(rollingMean(k, smoothD), length + smoothD);

  const result = [k, d] as StochasticResult;
  result.k = k;
  result.d = d;
  return result;
}

/**
 * Williams %R - Vectorized.
 */
export function williamsR(
  close: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  length = 14
): Float64Array {
  const c = ensureArray(close);
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = c.length;

  const hh = rollingMax(h, length);
  const ll = rollingMin(l, length);
  const wr = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const denom = hh[i] - ll[i] === 0 ? 1e-10 : hh[i] - ll[i];
    wr[i] = (-100 * (hh[i] - c[i])) / denom;
  }

  return padNan(wr, length);
}

/**
 * Commodity Channel Index (CCI) - Vectorized.
 */
export function cci(
  close: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  length = 20
): Float64Array {
  const c = ensureArray(close);
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = c.length;

  const tp = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    tp[i] = (h[i] + l[i] + c[i]) / 3;
  }

  const smaTp = rollingMean(tp, length);
  const cciVals = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < length - 1) {
      cciVals[i] = NaN;
      continue;
    }

    const mean = smaTp[i];
    let devSum = 0;
    for (let j = i - length + 1; j <= i; j++) {
      devSum += Math.abs(tp[j] - mean);
    }
    const meanDev = devSum / length === 0 ? 1e-10 : devSum / length;
    cciVals[i] = (tp[i] - mean) / (0.015 * meanDev);
  }

  return padNan(cciVals, length);
}

/**
 * Rate of Change (ROC) - Vectorized.
 */
export function roc(source: ArrayLike, length = 12): Float64Array {
  const close = ensureArray(source);
  const prev = shift(close, length);
  const n = close.length;
  const rocVals = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < length || Number.isNaN(prev[i])) {
      rocVals[i] = NaN;
    } else {
      const p = prev[i] === 0 ? 1e-10 : prev[i];
      rocVals[i] = (100 * (close[i] - p)) / p;
    }
  }

  return padNan(rocVals, length);
}

/**
 * Momentum Indicator - Vectorized.
 */
export function momentum(source: ArrayLike, length = 10): Float64Array {
  const close = ensureArray(source);
  const prev = shift(close, length);
  const n = close.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    result[i] = close[i] - prev[i];
  }
  return result;
}

/**
 * Average Directional Index (ADX) - Vectorized.
 * Returns: [adx, plusDi, minusDi] & { adx, plusDi, minusDi }
 * Warmup: length * 2 bars
 */
export function adx(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  length = 14
): AdxResult {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = h.length;

  const tr = trueRange(h, l, c);
  const prevH = shift(h, 1);
  const prevL = shift(l, 1);

  const plusDm = new Float64Array(n);
  const minusDm = new Float64Array(n);

  for (let i = 1; i < n; i++) {
    const upMove = h[i] - prevH[i];
    const downMove = prevL[i] - l[i];

    if (upMove > downMove && upMove > 0) {
      plusDm[i] = upMove;
    }
    if (downMove > upMove && downMove > 0) {
      minusDm[i] = downMove;
    }
  }

  const atrVals = wilderSmooth(tr, length, 0);
  const smoothPlusDm = wilderSmooth(plusDm, length, 1);
  const smoothMinusDm = wilderSmooth(minusDm, length, 1);

  const plusDi = new Float64Array(n);
  const minusDi = new Float64Array(n);
  const dx = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const atrSafe = atrVals[i] === 0 ? 1e-10 : atrVals[i];
    plusDi[i] = (100 * smoothPlusDm[i]) / atrSafe;
    minusDi[i] = (100 * smoothMinusDm[i]) / atrSafe;

    const diSum = plusDi[i] + minusDi[i] === 0 ? 1e-10 : plusDi[i] + minusDi[i];
    dx[i] = (100 * Math.abs(plusDi[i] - minusDi[i])) / diSum;
  }

  const adxVal = wilderSmooth(dx, length, length);

  padNan(adxVal, length * 2);
  padNan(plusDi, length);
  padNan(minusDi, length);

  const result = [adxVal, plusDi, minusDi] as AdxResult;
  result.adx = adxVal;
  result.plusDi = plusDi;
  result.minusDi = minusDi;
  return result;
}

/**
 * Stochastic RSI - Vectorized.
 * Returns: [k, d] & { k, d }
 */
export function stochRsi(
  source: ArrayLike,
  rsiLength = 14,
  stochLength = 14,
  smoothK = 3,
  smoothD = 3
): StochRsiResult {
  const rsiVals = rsi(source, rsiLength);
  const n = rsiVals.length;

  const hh = rollingMax(rsiVals, stochLength);
  const ll = rollingMin(rsiVals, stochLength);
  const stoch = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const denom = hh[i] - ll[i] === 0 ? 1e-10 : hh[i] - ll[i];
    stoch[i] = (rsiVals[i] - ll[i]) / denom;
  }

  const k = rollingMean(stoch, smoothK);
  const d = rollingMean(k, smoothD);
  const warmup = rsiLength + stochLength;

  padNan(k, warmup);
  padNan(d, warmup + smoothD);

  const result = [k, d] as StochRsiResult;
  result.k = k;
  result.d = d;
  return result;
}

/**
 * TRIX Indicator - Vectorized.
 */
export function trix(source: ArrayLike, length = 15): Float64Array {
  const close = ensureArray(source);
  const ema1 = ewmMean(close, length);
  const ema2 = ewmMean(ema1, length);
  const ema3 = ewmMean(ema2, length);
  const prevEma3 = shift(ema3, 1);
  const n = close.length;
  const trixVals = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const p = prevEma3[i] === 0 ? 1e-10 : prevEma3[i];
    trixVals[i] = (100 * (ema3[i] - p)) / p;
  }

  return padNan(trixVals, length * 3);
}

/**
 * Ultimate Oscillator - Vectorized.
 */
export function ultimateOscillator(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  s1 = 7,
  s2 = 14,
  s3 = 28
): Float64Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const prevC = shift(c, 1);
  const n = c.length;

  const bp = new Float64Array(n);
  const tr = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const pc = Number.isNaN(prevC[i]) ? c[i] : prevC[i];
    bp[i] = c[i] - Math.min(l[i], pc);
    tr[i] = Math.max(h[i], pc) - Math.min(l[i], pc);
  }

  const sumBpS1 = rollingSum(bp, s1);
  const sumTrS1 = rollingSum(tr, s1);
  const sumBpS2 = rollingSum(bp, s2);
  const sumTrS2 = rollingSum(tr, s2);
  const sumBpS3 = rollingSum(bp, s3);
  const sumTrS3 = rollingSum(tr, s3);

  const uo = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const tr1 = sumTrS1[i] === 0 ? 1e-10 : sumTrS1[i];
    const tr2 = sumTrS2[i] === 0 ? 1e-10 : sumTrS2[i];
    const tr3 = sumTrS3[i] === 0 ? 1e-10 : sumTrS3[i];

    uo[i] = 100 * (
      (4 * sumBpS1[i]) / tr1 +
      (2 * sumBpS2[i]) / tr2 +
      sumBpS3[i] / tr3
    ) / 7;
  }

  return padNan(uo, s3);
}

/**
 * Chande Momentum Oscillator (CMO) - Vectorized.
 */
export function cmo(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const delta = diff(close, 1);
  const n = close.length;

  const up = new Float64Array(n);
  const down = new Float64Array(n);

  for (let i = 1; i < n; i++) {
    const d = delta[i];
    if (d > 0) up[i] = d;
    else if (d < 0) down[i] = -d;
  }

  const sumUp = rollingSum(up, length);
  const sumDown = rollingSum(down, length);
  const cmoVals = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const total = sumUp[i] + sumDown[i] === 0 ? 1e-10 : sumUp[i] + sumDown[i];
    cmoVals[i] = (100 * (sumUp[i] - sumDown[i])) / total;
  }

  return padNan(cmoVals, length);
}

/**
 * True Strength Index (TSI) - Vectorized.
 */
export function tsi(
  source: ArrayLike,
  longLength = 25,
  shortLength = 13
): Float64Array {
  const close = ensureArray(source);
  const mom = diff(close, 1);
  const n = close.length;

  const absMom = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (Number.isNaN(mom[i])) mom[i] = 0;
    absMom[i] = Math.abs(mom[i]);
  }

  const ema1 = ewmMean(mom, shortLength);
  const ema2 = ewmMean(ema1, longLength);

  const absEma1 = ewmMean(absMom, shortLength);
  const absEma2 = ewmMean(absEma1, longLength);

  const tsiVals = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const denom = absEma2[i] === 0 ? 1e-10 : absEma2[i];
    tsiVals[i] = (100 * ema2[i]) / denom;
  }

  return padNan(tsiVals, longLength + shortLength);
}
