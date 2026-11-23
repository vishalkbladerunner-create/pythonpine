/**
 * PythonPine Trend - High-Performance Vectorized Indicators
 */

import {
  ArrayLike,
  ensureArray,
  rollingMean,
  rollingMax,
  rollingMin,
  ewmMean,
  padNan,
  shift
} from './utils.js';
import { atr } from './volatility.js';

export type SupertrendResult = [Int32Array, Float64Array] & {
  trend: Int32Array;
  supertrend: Float64Array;
};

export type AroonResult = [Float64Array, Float64Array] & {
  aroonUp: Float64Array;
  aroonDown: Float64Array;
};

export type IchimokuResult = [
  Float64Array,
  Float64Array,
  Float64Array,
  Float64Array,
  Float64Array
] & {
  conversion: Float64Array;
  base: Float64Array;
  spanA: Float64Array;
  spanB: Float64Array;
  lagging: Float64Array;
};

/**
 * Simple Moving Average (SMA) - Vectorized.
 */
export function sma(source: ArrayLike, length = 14): Float64Array {
  return rollingMean(source, length);
}

/**
 * Exponential Moving Average (EMA) - Vectorized.
 */
export function ema(source: ArrayLike, length = 14): Float64Array {
  return ewmMean(source, length);
}

/**
 * Weighted Moving Average (WMA) - Vectorized.
 */
export function wma(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const n = close.length;
  const result = new Float64Array(n);
  const weightSum = (length * (length + 1)) / 2;

  for (let i = 0; i < n; i++) {
    if (i < length - 1) {
      result[i] = NaN;
      continue;
    }

    let sum = 0;
    for (let k = 0; k < length; k++) {
      const weight = k + 1;
      sum += close[i - length + 1 + k] * weight;
    }
    result[i] = sum / weightSum;
  }
  return result;
}

/**
 * Double EMA (DEMA) - Vectorized.
 */
export function dema(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const ema1 = ewmMean(close, length);
  const ema2 = ewmMean(ema1, length);
  const n = close.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    result[i] = 2 * ema1[i] - ema2[i];
  }
  return result;
}

/**
 * Triple EMA (TEMA) - Vectorized.
 */
export function tema(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const ema1 = ewmMean(close, length);
  const ema2 = ewmMean(ema1, length);
  const ema3 = ewmMean(ema2, length);
  const n = close.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    result[i] = 3 * ema1[i] - 3 * ema2[i] + ema3[i];
  }
  return result;
}

/**
 * Hull Moving Average (HMA) - Vectorized.
 */
export function hma(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const halfLength = Math.max(1, Math.floor(length / 2));
  const sqrtLength = Math.max(1, Math.floor(Math.sqrt(length)));

  const wmaHalf = wma(close, halfLength);
  const wmaFull = wma(close, length);
  const n = close.length;
  const diff = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    diff[i] = 2 * wmaHalf[i] - wmaFull[i];
  }

  return wma(diff, sqrtLength);
}

/**
 * Volume Weighted Moving Average (VWMA) - Vectorized.
 */
export function vwma(source: ArrayLike, volume: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const vol = ensureArray(volume);
  const n = close.length;

  const pv = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    pv[i] = close[i] * vol[i];
  }

  const sumPv = rollingMean(pv, length);
  const sumVol = rollingMean(vol, length);
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const sv = sumVol[i] === 0 ? 1e-10 : sumVol[i];
    result[i] = sumPv[i] / sv;
  }
  return result;
}

/**
 * Kaufman Adaptive Moving Average (KAMA) - Vectorized.
 */
export function kama(
  source: ArrayLike,
  length = 10,
  fast = 2,
  slow = 30
): Float64Array {
  const close = ensureArray(source);
  const n = close.length;
  const fastSc = 2 / (fast + 1);
  const slowSc = 2 / (slow + 1);

  const kamaVals = new Float64Array(n);
  if (n === 0) return kamaVals;

  kamaVals[0] = close[0];

  for (let i = 1; i < n; i++) {
    if (i < length) {
      kamaVals[i] = close[i];
    } else {
      const change = Math.abs(close[i] - close[i - length]);
      let volatility = 0;
      for (let j = i - length + 1; j <= i; j++) {
        volatility += Math.abs(close[j] - close[j - 1]);
      }
      volatility = Math.max(volatility, 1e-10);

      const er = change / volatility;
      const sc = Math.pow(er * (fastSc - slowSc) + slowSc, 2);
      kamaVals[i] = kamaVals[i - 1] + sc * (close[i] - kamaVals[i - 1]);
    }
  }

  return padNan(kamaVals, length);
}

/**
 * SuperTrend - Vectorized.
 * Returns: [trend, supertrend] & { trend, supertrend }
 * trend: 1 = uptrend, -1 = downtrend
 */
export function supertrend(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  length = 10,
  mult = 3.0
): SupertrendResult {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = c.length;

  const atrVals = atr(h, l, c, length);
  const upper = new Float64Array(n);
  const lower = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const hl2 = (h[i] + l[i]) / 2;
    upper[i] = hl2 + mult * atrVals[i];
    lower[i] = hl2 - mult * atrVals[i];
  }

  const trend = new Int32Array(n).fill(1);
  const stLine = new Float64Array(n);
  stLine[0] = lower[0];

  for (let i = 1; i < n; i++) {
    if (c[i] > upper[i - 1]) {
      trend[i] = 1;
    } else if (c[i] < lower[i - 1]) {
      trend[i] = -1;
    } else {
      trend[i] = trend[i - 1];
    }

    if (trend[i] === 1) {
      stLine[i] = trend[i - 1] === 1 && !Number.isNaN(stLine[i - 1])
        ? Math.max(lower[i], stLine[i - 1])
        : lower[i];
    } else {
      stLine[i] = trend[i - 1] === -1 && !Number.isNaN(stLine[i - 1])
        ? Math.min(upper[i], stLine[i - 1])
        : upper[i];
    }
  }

  padNan(stLine, length);

  const result = [trend, stLine] as SupertrendResult;
  result.trend = trend;
  result.supertrend = stLine;
  return result;
}

/**
 * Aroon Indicator - Vectorized.
 * Returns: [aroonUp, aroonDown] & { aroonUp, aroonDown }
 */
export function aroon(
  high: ArrayLike,
  low: ArrayLike,
  length = 14
): AroonResult {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;

  const aroonUp = new Float64Array(n).fill(NaN);
  const aroonDown = new Float64Array(n).fill(NaN);

  for (let i = length; i < n; i++) {
    let maxVal = -Infinity;
    let maxIdx = -1;
    let minVal = Infinity;
    let minIdx = -1;

    for (let j = i - length + 1; j <= i; j++) {
      if (h[j] >= maxVal) {
        maxVal = h[j];
        maxIdx = j;
      }
      if (l[j] <= minVal) {
        minVal = l[j];
        minIdx = j;
      }
    }

    const barsSinceHigh = i - maxIdx;
    const barsSinceLow = i - minIdx;

    aroonUp[i] = (100 * (length - barsSinceHigh)) / length;
    aroonDown[i] = (100 * (length - barsSinceLow)) / length;
  }

  const result = [aroonUp, aroonDown] as AroonResult;
  result.aroonUp = aroonUp;
  result.aroonDown = aroonDown;
  return result;
}

/**
 * Ichimoku Cloud - Vectorized.
 * Returns: [conversion, base, spanA, spanB, lagging] & { conversion, base, spanA, spanB, lagging }
 */
export function ichimoku(
  high: ArrayLike,
  low: ArrayLike,
  close: ArrayLike,
  tenkan = 9,
  kijun = 26,
  senkou = 52
): IchimokuResult {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = c.length;

  const maxTenkan = rollingMax(h, tenkan);
  const minTenkan = rollingMin(l, tenkan);
  const maxKijun = rollingMax(h, kijun);
  const minKijun = rollingMin(l, kijun);
  const maxSenkou = rollingMax(h, senkou);
  const minSenkou = rollingMin(l, senkou);

  const conversion = new Float64Array(n);
  const base = new Float64Array(n);
  const spanA = new Float64Array(n);
  const spanB = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    conversion[i] = (maxTenkan[i] + minTenkan[i]) / 2;
    base[i] = (maxKijun[i] + minKijun[i]) / 2;
    spanA[i] = (conversion[i] + base[i]) / 2;
    spanB[i] = (maxSenkou[i] + minSenkou[i]) / 2;
  }

  const lagging = shift(c, -kijun);

  const result = [conversion, base, spanA, spanB, lagging] as IchimokuResult;
  result.conversion = conversion;
  result.base = base;
  result.spanA = spanA;
  result.spanB = spanB;
  result.lagging = lagging;
  return result;
}

/**
 * Linear Regression Line - Vectorized.
 */
export function linearRegression(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const n = close.length;
  const result = new Float64Array(n);

  let sumX = 0;
  let sumX2 = 0;
  for (let i = 0; i < length; i++) {
    sumX += i;
    sumX2 += i * i;
  }

  for (let i = 0; i < n; i++) {
    if (i < length - 1) {
      result[i] = NaN;
      continue;
    }

    let sumY = 0;
    let sumXY = 0;
    for (let k = 0; k < length; k++) {
      const y = close[i - length + 1 + k];
      sumY += y;
      sumXY += k * y;
    }

    const denom = length * sumX2 - sumX * sumX + 1e-10;
    const slope = (length * sumXY - sumX * sumY) / denom;
    const intercept = (sumY - slope * sumX) / length;

    result[i] = slope * (length - 1) + intercept;
  }

  return result;
}

/**
 * Parabolic SAR - Optimized.
 */
export function parabolicSar(
  high: ArrayLike,
  low: ArrayLike,
  afStep = 0.02,
  afMax = 0.2
): Float64Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const n = h.length;

  const sar = new Float64Array(n);
  if (n === 0) return sar;

  let trend = 1;
  let af = afStep;
  let ep = h[0];
  sar[0] = l[0];

  for (let i = 1; i < n; i++) {
    const prevSar = sar[i - 1];

    if (trend === 1) {
      sar[i] = prevSar + af * (ep - prevSar);
      sar[i] = Math.min(sar[i], l[i - 1]);
      if (i > 1) {
        sar[i] = Math.min(sar[i], l[i - 2]);
      }

      if (l[i] < sar[i]) {
        trend = -1;
        sar[i] = ep;
        ep = l[i];
        af = afStep;
      } else {
        if (h[i] > ep) {
          ep = h[i];
          af = Math.min(af + afStep, afMax);
        }
      }
    } else {
      sar[i] = prevSar + af * (ep - prevSar);
      sar[i] = Math.max(sar[i], h[i - 1]);
      if (i > 1) {
        sar[i] = Math.max(sar[i], h[i - 2]);
      }

      if (h[i] > sar[i]) {
        trend = 1;
        sar[i] = ep;
        ep = h[i];
        af = afStep;
      } else {
        if (l[i] < ep) {
          ep = l[i];
          af = Math.min(af + afStep, afMax);
        }
      }
    }
  }

  return sar;
}

/**
 * Zero-Lag EMA (ZLEMA) - Vectorized.
 */
export function zeroLagEma(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const ema1 = ewmMean(close, length);
  const ema2 = ewmMean(ema1, length);
  const n = close.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    result[i] = 2 * ema1[i] - ema2[i];
  }
  return result;
}

/**
 * McGinley Dynamic - Optimized.
 */
export function mcginleyDynamic(source: ArrayLike, length = 14): Float64Array {
  const close = ensureArray(source);
  const n = close.length;
  const md = new Float64Array(n);
  if (n === 0) return md;

  md[0] = close[0];

  for (let i = 1; i < n; i++) {
    const ratio = md[i - 1] !== 0 ? close[i] / md[i - 1] : 1;
    const ratio4 = Math.pow(ratio, 4);
    md[i] = md[i - 1] + (close[i] - md[i - 1]) / (length * ratio4);
  }

  return md;
}
