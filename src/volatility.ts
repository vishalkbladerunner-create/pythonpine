/**
 * PythonPine Volatility - High-Performance Vectorized Indicators
 */

import {
  ArrayLike,
  ensureArray,
  rollingMean,
  rollingStd,
  rollingMax,
  rollingMin,
  ewmMean,
  wilderSmooth,
  trueRange,
  padNan,
  shift
} from './utils.js';

export type BollingerBandsResult = [Float64Array, Float64Array, Float64Array] & {
  upper: Float64Array;
  lower: Float64Array;
  middle: Float64Array;
};

export type KeltnerChannelResult = [Float64Array, Float64Array, Float64Array] & {
  upper: Float64Array;
  lower: Float64Array;
  middle: Float64Array;
};

export type DonchianChannelResult = [Float64Array, Float64Array, Float64Array] & {
  upper: Float64Array;
  lower: Float64Array;
  middle: Float64Array;
};

/**
 * Average True Range (ATR) - Vectorized with Wilder's smoothing.
 * Warmup: length - 1 bars (Pine Script parity)
 */
export function atr(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  length = 14
): Float64Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);

  const tr = trueRange(h, l, c);
  const atrVals = wilderSmooth(tr, length, 0);
  return padNan(atrVals, length - 1);
}

/**
 * Bollinger Bands - Vectorized.
 * Returns: [upper, lower, middle] & { upper, lower, middle }
 */
export function bollingerBands(
  source: ArrayLike,
  length = 20,
  mult = 2.0
): BollingerBandsResult {
  const close = ensureArray(source);
  const n = close.length;

  const middle = rollingMean(close, length);
  const std = rollingStd(close, length, 0);

  const upper = new Float64Array(n);
  const lower = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    upper[i] = middle[i] + mult * std[i];
    lower[i] = middle[i] - mult * std[i];
  }

  padNan(upper, length);
  padNan(lower, length);
  padNan(middle, length);

  const result = [upper, lower, middle] as BollingerBandsResult;
  result.upper = upper;
  result.lower = lower;
  result.middle = middle;
  return result;
}

/**
 * Keltner Channel - Vectorized.
 * Returns: [upper, lower, middle] & { upper, lower, middle }
 */
export function keltnerChannel(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  length = 20,
  mult = 2.0
): KeltnerChannelResult {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = c.length;

  const tp = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    tp[i] = (h[i] + l[i] + c[i]) / 3;
  }

  const middle = ewmMean(tp, length);
  const atrVals = atr(h, l, c, length);

  const upper = new Float64Array(n);
  const lower = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    upper[i] = middle[i] + mult * atrVals[i];
    lower[i] = middle[i] - mult * atrVals[i];
  }

  const result = [upper, lower, middle] as KeltnerChannelResult;
  result.upper = upper;
  result.lower = lower;
  result.middle = middle;
  return result;
}

/**
 * Donchian Channel - Vectorized.
 * Returns: [upper, lower, middle] & { upper, lower, middle }
 */
export function donchianChannel(
  high: ArrayLike,
  low: ArrayLike,
  length = 20
): DonchianChannelResult {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;

  const upper = rollingMax(h, length);
  const lower = rollingMin(l, length);
  const middle = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    middle[i] = (upper[i] + lower[i]) / 2;
  }

  padNan(upper, length);
  padNan(lower, length);
  padNan(middle, length);

  const result = [upper, lower, middle] as DonchianChannelResult;
  result.upper = upper;
  result.lower = lower;
  result.middle = middle;
  return result;
}

/**
 * Bollinger Bandwidth - Vectorized.
 */
export function bollingerBandwidth(
  source: ArrayLike,
  length = 20,
  mult = 2.0
): Float64Array {
  const { upper, lower, middle } = bollingerBands(source, length, mult);
  const n = upper.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const m = middle[i] === 0 ? 1e-10 : middle[i];
    result[i] = (upper[i] - lower[i]) / m;
  }
  return result;
}

/**
 * Bollinger %B - Vectorized.
 */
export function bollingerPercentB(
  source: ArrayLike,
  length = 20,
  mult = 2.0
): Float64Array {
  const close = ensureArray(source);
  const { upper, lower } = bollingerBands(close, length, mult);
  const n = close.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const denom = upper[i] - lower[i] === 0 ? 1e-10 : upper[i] - lower[i];
    result[i] = (close[i] - lower[i]) / denom;
  }
  return result;
}

/**
 * Historical Volatility (HV) - Vectorized.
 */
export function historicalVolatility(
  source: ArrayLike,
  length = 20,
  annualize = true
): Float64Array {
  const close = ensureArray(source);
  const n = close.length;
  const logReturns = new Float64Array(n);
  logReturns[0] = NaN;

  for (let i = 1; i < n; i++) {
    logReturns[i] = Math.log(close[i] / close[i - 1]);
  }

  const hv = rollingStd(logReturns, length, 1);
  if (annualize) {
    const factor = Math.sqrt(252);
    for (let i = 0; i < n; i++) {
      hv[i] *= factor;
    }
  }

  return padNan(hv, length);
}

/**
 * NATR (Normalized ATR) - Vectorized.
 */
export function normalizedAtr(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  length = 14
): Float64Array {
  const c = ensureArray(close);
  const atrVals = atr(high, low, close, length);
  const n = c.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const denom = c[i] === 0 ? 1e-10 : c[i];
    result[i] = (100 * atrVals[i]) / denom;
  }
  return result;
}

/**
 * Ulcer Index - Vectorized.
 */
export function ulcerIndex(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const maxClose = rollingMax(close, length);
  const n = close.length;
  const squaredDd = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const mc = maxClose[i] === 0 ? 1e-10 : maxClose[i];
    const dd = (100 * (close[i] - mc)) / mc;
    squaredDd[i] = dd * dd;
  }

  const meanSq = rollingMean(squaredDd, length);
  const ui = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    ui[i] = Math.sqrt(meanSq[i]);
  }
  return padNan(ui, length);
}

/**
 * Chaikin Volatility - Vectorized.
 */
export function chaikinVolatility(
  high: ArrayLike,
  low: ArrayLike,
  length = 10
): Float64Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;

  const hlDiff = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    hlDiff[i] = h[i] - l[i];
  }

  const emaShort = ewmMean(hlDiff, length);
  const shifted = shift(emaShort, length);
  const cv = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const s = shifted[i] === 0 ? 1e-10 : shifted[i];
    cv[i] = (100 * (emaShort[i] - s)) / s;
  }

  return padNan(cv, length * 2);
}

/**
 * Mass Index - Vectorized.
 */
export function massIndex(
  high: ArrayLike,
  low: ArrayLike,
  emaLength = 9,
  sumLength = 25
): Float64Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;

  const hlRange = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    hlRange[i] = h[i] - l[i];
  }

  const ema1 = ewmMean(hlRange, emaLength);
  const ema2 = ewmMean(ema1, emaLength);
  const ratio = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const denom = ema2[i] === 0 ? 1e-10 : ema2[i];
    ratio[i] = ema1[i] / denom;
  }

  const mass = new Float64Array(n);
  let currentSum = 0;
  for (let i = 0; i < n; i++) {
    currentSum += ratio[i];
    if (i >= sumLength) {
      currentSum -= ratio[i - sumLength];
    }
    mass[i] = i >= sumLength - 1 ? currentSum : NaN;
  }

  return padNan(mass, sumLength + emaLength);
}

/**
 * Parkinson Volatility Estimator - Vectorized.
 */
export function parkinsonVolatility(
  high: ArrayLike,
  low: ArrayLike,
  length = 10
): Float64Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;

  const logHl = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ratio = h[i] / (l[i] === 0 ? 1e-10 : l[i]);
    const val = Math.log(ratio);
    logHl[i] = val * val;
  }

  const factor = 1 / (4 * Math.log(2));
  const meanLog = rollingMean(logHl, length);
  const pv = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    pv[i] = Math.sqrt(factor * meanLog[i]);
  }
  return padNan(pv, length);
}

/**
 * Garman-Klass Volatility Estimator - Vectorized.
 */
export function garmanKlassVolatility(
  open_: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  length = 10
): Float64Array {
  const o = ensureArray(open_);
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = o.length;

  const gk = new Float64Array(n);
  const c2 = 2 * Math.log(2) - 1;

  for (let i = 0; i < n; i++) {
    const oSafe = o[i] === 0 ? 1e-10 : o[i];
    const lSafe = l[i] === 0 ? 1e-10 : l[i];

    const logHl = Math.log(h[i] / lSafe);
    const logCo = Math.log(c[i] / oSafe);

    gk[i] = 0.5 * logHl * logHl - c2 * logCo * logCo;
  }

  const meanGk = rollingMean(gk, length);
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    result[i] = Math.sqrt(Math.max(0, meanGk[i]));
  }
  return padNan(result, length);
}
