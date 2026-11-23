/**
 * PythonPine Price Action - Vectorized Candlestick & Level Detection
 */

import {
  ArrayLike,
  ensureArray,
  shift,
  rollingMax,
  rollingMin
} from './utils.js';

export type SupportResistanceResult = [Float64Array, Float64Array] & {
  support: Float64Array;
  resistance: Float64Array;
};

export type HigherHighsLowerLowsResult = [Int32Array, Int32Array] & {
  higherHigh: Int32Array;
  lowerLow: Int32Array;
};

/**
 * Engulfing Pattern Detection - Vectorized.
 * +1: Bullish engulfing
 * -1: Bearish engulfing
 *  0: No pattern
 */
export function engulfing(
  open_: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike
): Int32Array {
  const o = ensureArray(open_);
  const c = ensureArray(close);
  const n = c.length;
  const result = new Int32Array(n);

  const prevO = shift(o, 1);
  const prevC = shift(c, 1);

  for (let i = 1; i < n; i++) {
    const isPrevBearish = prevC[i] < prevO[i];
    const isCurrentBullish = c[i] > o[i];
    const isBullishEngulf = isPrevBearish && isCurrentBullish && o[i] <= prevC[i] && c[i] >= prevO[i];

    const isPrevBullish = prevC[i] > prevO[i];
    const isCurrentBearish = c[i] < o[i];
    const isBearishEngulf = isPrevBullish && isCurrentBearish && o[i] >= prevC[i] && c[i] <= prevO[i];

    if (isBullishEngulf) {
      result[i] = 1;
    } else if (isBearishEngulf) {
      result[i] = -1;
    }
  }

  return result;
}

/**
 * Doji Detection - Vectorized.
 * 1: Doji detected
 * 0: No doji
 */
export function doji(
  open_: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  threshold = 0.1
): Int32Array {
  const o = ensureArray(open_);
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = c.length;
  const result = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    const body = Math.abs(c[i] - o[i]);
    const rangeVal = h[i] - l[i] === 0 ? 1e-10 : h[i] - l[i];
    if (body / rangeVal < threshold) {
      result[i] = 1;
    }
  }

  return result;
}

/**
 * Hammer Pattern (Bullish) - Vectorized.
 */
export function hammer(
  open_: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  bodyRatio = 0.3,
  wickRatio = 2.0
): Int32Array {
  const o = ensureArray(open_);
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = c.length;
  const result = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    const body = Math.abs(c[i] - o[i]);
    const rangeVal = h[i] - l[i] === 0 ? 1e-10 : h[i] - l[i];

    const bodyTop = Math.max(c[i], o[i]);
    const bodyBottom = Math.min(c[i], o[i]);
    const upperWick = h[i] - bodyTop;
    const lowerWick = bodyBottom - l[i];

    const isSmallBody = body / rangeVal < bodyRatio;
    const isLongLower = lowerWick > wickRatio * body;
    const isSmallUpper = upperWick < body;

    if (isSmallBody && isLongLower && isSmallUpper) {
      result[i] = 1;
    }
  }

  return result;
}

/**
 * Shooting Star Pattern (Bearish) - Vectorized.
 */
export function shootingStar(
  open_: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  bodyRatio = 0.3,
  wickRatio = 2.0
): Int32Array {
  const o = ensureArray(open_);
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = c.length;
  const result = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    const body = Math.abs(c[i] - o[i]);
    const rangeVal = h[i] - l[i] === 0 ? 1e-10 : h[i] - l[i];

    const bodyTop = Math.max(c[i], o[i]);
    const bodyBottom = Math.min(c[i], o[i]);
    const upperWick = h[i] - bodyTop;
    const lowerWick = bodyBottom - l[i];

    const isSmallBody = body / rangeVal < bodyRatio;
    const isLongUpper = upperWick > wickRatio * body;
    const isSmallLower = lowerWick < body;

    if (isSmallBody && isLongUpper && isSmallLower) {
      result[i] = 1;
    }
  }

  return result;
}

/**
 * Morning Star (3-bar bullish reversal) - Vectorized.
 */
export function morningStar(
  open_: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike
): Int32Array {
  const o = ensureArray(open_);
  const c = ensureArray(close);
  const n = c.length;
  const result = new Int32Array(n);

  for (let i = 2; i < n; i++) {
    const o2 = o[i - 2];
    const c2 = c[i - 2];
    const o1 = o[i - 1];
    const c1 = c[i - 1];
    const o0 = o[i];
    const c0 = c[i];

    const prev2Bearish = c2 < o2 && Math.abs(c2 - o2) > 0;
    const prev1Body = Math.abs(c1 - o1);
    const prev2Body = Math.abs(c2 - o2);
    const smallBody = prev1Body < prev2Body * 0.5;
    const currentBullish = c0 > o0;
    const closesAboveMid = c0 > (o2 + c2) / 2;

    if (prev2Bearish && smallBody && currentBullish && closesAboveMid) {
      result[i] = 1;
    }
  }

  return result;
}

/**
 * Evening Star (3-bar bearish reversal) - Vectorized.
 */
export function eveningStar(
  open_: ArrayLike,
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike
): Int32Array {
  const o = ensureArray(open_);
  const c = ensureArray(close);
  const n = c.length;
  const result = new Int32Array(n);

  for (let i = 2; i < n; i++) {
    const o2 = o[i - 2];
    const c2 = c[i - 2];
    const o1 = o[i - 1];
    const c1 = c[i - 1];
    const o0 = o[i];
    const c0 = c[i];

    const prev2Bullish = c2 > o2;
    const prev1Body = Math.abs(c1 - o1);
    const prev2Body = Math.abs(c2 - o2);
    const smallBody = prev1Body < prev2Body * 0.5;
    const currentBearish = c0 < o0;
    const closesBelowMid = c0 < (o2 + c2) / 2;

    if (prev2Bullish && smallBody && currentBearish && closesBelowMid) {
      result[i] = 1;
    }
  }

  return result;
}

/**
 * Inside Bar Detection - Vectorized.
 */
export function insideBar(high: ArrayLike, low: ArrayLike): Int32Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;
  const result = new Int32Array(n);

  for (let i = 1; i < n; i++) {
    if (h[i] < h[i - 1] && l[i] > l[i - 1]) {
      result[i] = 1;
    }
  }
  return result;
}

/**
 * Outside Bar Detection - Vectorized.
 */
export function outsideBar(high: ArrayLike, low: ArrayLike): Int32Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;
  const result = new Int32Array(n);

  for (let i = 1; i < n; i++) {
    if (h[i] > h[i - 1] && l[i] < l[i - 1]) {
      result[i] = 1;
    }
  }
  return result;
}

/**
 * Support/Resistance Zone Detection based on volume profile.
 * Returns: [support, resistance] & { support, resistance }
 */
export function supportResistanceZones(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  volume: ArrayLike,
  lookback = 50,
  numZones = 3
): SupportResistanceResult {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const v = ensureArray(volume);
  const n = c.length;

  const support = new Float64Array(n).fill(NaN);
  const resistance = new Float64Array(n).fill(NaN);

  for (let i = lookback; i < n; i++) {
    let priceMin = Infinity;
    let priceMax = -Infinity;

    for (let j = i - lookback; j < i; j++) {
      if (l[j] < priceMin) priceMin = l[j];
      if (h[j] > priceMax) priceMax = h[j];
    }

    if (priceMax === priceMin) continue;

    const numBins = 20;
    const binStep = (priceMax - priceMin) / (numBins - 1);
    const volumeProfile = new Float64Array(numBins - 1);

    for (let j = i - lookback; j < i; j++) {
      const price = c[j];
      let binIdx = Math.floor((price - priceMin) / binStep);
      binIdx = Math.max(0, Math.min(numBins - 2, binIdx));
      volumeProfile[binIdx] += v[j];
    }

    // Sort bin indices descending by volume
    const sortedIndices = Array.from({ length: numBins - 1 }, (_, idx) => idx)
      .sort((a, b) => volumeProfile[b] - volumeProfile[a]);

    const currentPrice = c[i];
    const zonePrices: number[] = [];
    for (let k = 0; k < Math.min(numZones * 2, sortedIndices.length); k++) {
      const idx = sortedIndices[k];
      const binStart = priceMin + idx * binStep;
      const binEnd = binStart + binStep;
      zonePrices.push((binStart + binEnd) / 2);
    }

    let maxBelow = -Infinity;
    let minAbove = Infinity;

    for (const p of zonePrices) {
      if (p < currentPrice && p > maxBelow) maxBelow = p;
      if (p > currentPrice && p < minAbove) minAbove = p;
    }

    if (maxBelow !== -Infinity) support[i] = maxBelow;
    if (minAbove !== Infinity) resistance[i] = minAbove;
  }

  const result = [support, resistance] as SupportResistanceResult;
  result.support = support;
  result.resistance = resistance;
  return result;
}

/**
 * Higher Highs / Lower Lows Detection - Vectorized.
 * Returns: [higherHigh, lowerLow] & { higherHigh, lowerLow }
 */
export function higherHighsLowerLows(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  lookback = 5
): HigherHighsLowerLowsResult {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;

  const prevHigh = rollingMax(shift(h, 1), lookback);
  const prevLow = rollingMin(shift(l, 1), lookback);

  const hh = new Int32Array(n);
  const ll = new Int32Array(n);

  for (let i = lookback; i < n; i++) {
    if (h[i] > prevHigh[i]) hh[i] = 1;
    if (l[i] < prevLow[i]) ll[i] = 1;
  }

  const result = [hh, ll] as HigherHighsLowerLowsResult;
  result.higherHigh = hh;
  result.lowerLow = ll;
  return result;
}
