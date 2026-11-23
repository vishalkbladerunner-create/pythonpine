/**
 * PythonPine Normalize - ML Alpha Factor Preprocessing
 * Convert raw indicator values into ML-ready features.
 */

import {
  ArrayLike,
  ensureArray,
  rollingMean,
  rollingStd,
  rollingMin,
  rollingMax
} from './utils.js';

/**
 * Rolling Z-Score Normalization.
 * Formula: (value - rolling_mean) / rolling_std
 * Returns values centered around 0 with unit variance.
 */
export function zscore(data: ArrayLike, window = 20): Float64Array {
  const arr = ensureArray(data);
  const mean = rollingMean(arr, window);
  const std = rollingStd(arr, window, 0);
  const n = arr.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < window || Number.isNaN(mean[i]) || Number.isNaN(std[i])) {
      result[i] = NaN;
    } else {
      const s = std[i] === 0 ? 1e-10 : std[i];
      result[i] = (arr[i] - mean[i]) / s;
    }
  }

  return result;
}

/**
 * Rolling Percentile Rank.
 * Returns where the current value ranks (0-100) relative to the last N bars.
 */
export function percentileRank(data: ArrayLike, window = 100): Float64Array {
  const arr = ensureArray(data);
  const n = arr.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < window) {
      result[i] = NaN;
      continue;
    }

    const currentVal = arr[i];
    let countLess = 0;
    for (let j = i - window; j <= i; j++) {
      if (arr[j] < currentVal) {
        countLess++;
      }
    }
    result[i] = (countLess / window) * 100;
  }

  return result;
}

/**
 * Log Returns - makes price data stationary.
 * Formula: ln(price[t] / price[t-1])
 */
export function logReturns(data: ArrayLike): Float64Array {
  const arr = ensureArray(data);
  const n = arr.length;
  const result = new Float64Array(n);
  result[0] = NaN;

  for (let i = 1; i < n; i++) {
    result[i] = Math.log(arr[i] / arr[i - 1]);
  }
  return result;
}

/**
 * Simple percentage returns.
 * Formula: (price[t] - price[t-1]) / price[t-1]
 */
export function simpleReturns(data: ArrayLike): Float64Array {
  const arr = ensureArray(data);
  const n = arr.length;
  const result = new Float64Array(n);
  result[0] = NaN;

  for (let i = 1; i < n; i++) {
    const prev = arr[i - 1];
    result[i] = prev === 0 ? NaN : (arr[i] - prev) / prev;
  }
  return result;
}

/**
 * Fractional Differentiation (López de Prado method).
 * Balances stationarity vs memory preservation.
 */
export function fractionalDiff(
  data: ArrayLike,
  d = 0.4,
  threshold = 1e-5
): Float64Array {
  const arr = ensureArray(data);
  const n = arr.length;

  // Calculate weights
  const weights: number[] = [1.0];
  let k = 1;
  while (Math.abs(weights[weights.length - 1]) > threshold) {
    const prevW = weights[weights.length - 1];
    const w = -prevW * (d - k + 1) / k;
    weights.push(w);
    k++;
  }

  // Reverse weights for convolution
  const revWeights = new Float64Array(weights.reverse());
  const wLen = revWeights.length;
  const result = new Float64Array(n).fill(NaN);

  for (let i = wLen - 1; i < n; i++) {
    let dot = 0;
    for (let j = 0; j < wLen; j++) {
      dot += revWeights[j] * arr[i - wLen + 1 + j];
    }
    result[i] = dot;
  }

  return result;
}

/**
 * Rolling Min-Max Scaling to [0, 1] range.
 */
export function minmaxScale(data: ArrayLike, window = 20): Float64Array {
  const arr = ensureArray(data);
  const n = arr.length;
  const rollMin = rollingMin(arr, window);
  const rollMax = rollingMax(arr, window);
  const scaled = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < window - 1) {
      scaled[i] = NaN;
      continue;
    }
    const rangeVal = rollMax[i] - rollMin[i] === 0 ? 1e-10 : rollMax[i] - rollMin[i];
    scaled[i] = (arr[i] - rollMin[i]) / rangeVal;
  }

  return scaled;
}

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

/**
 * Robust Scaling using rolling median and IQR.
 * Formula: (value - median) / IQR
 */
export function robustScale(data: ArrayLike, window = 20): Float64Array {
  const arr = ensureArray(data);
  const n = arr.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < window - 1) {
      result[i] = NaN;
      continue;
    }

    const slice: number[] = [];
    for (let j = i - window + 1; j <= i; j++) {
      slice.push(arr[j]);
    }
    slice.sort((a, b) => a - b);

    const median = quantile(slice, 0.5);
    const q1 = quantile(slice, 0.25);
    const q3 = quantile(slice, 0.75);
    const iqr = q3 - q1 === 0 ? 1e-10 : q3 - q1;

    result[i] = (arr[i] - median) / iqr;
  }

  return result;
}

/**
 * CUSUM Filter - detects structural breaks.
 * Returns 1 when cumulative sum exceeds threshold, -1 for negative threshold.
 */
export function cusumFilter(data: ArrayLike, threshold: number): Float64Array {
  const arr = ensureArray(data);
  const n = arr.length;
  let sPos = 0.0;
  let sNeg = 0.0;
  const result = new Float64Array(n);

  for (let i = 1; i < n; i++) {
    const diffVal = arr[i] - arr[i - 1];
    sPos = Math.max(0, sPos + diffVal);
    sNeg = Math.min(0, sNeg + diffVal);

    if (sPos > threshold) {
      result[i] = 1;
      sPos = 0;
    } else if (sNeg < -threshold) {
      result[i] = -1;
      sNeg = 0;
    }
  }

  return result;
}

/**
 * Triple Barrier Labeling for ML.
 * +1: Hit take profit first
 * -1: Hit stop loss first
 *  0: Neither hit within maxBars (hold)
 */
export function tripleBarrierLabels(
  close: ArrayLike,
  takeProfit: number,
  stopLoss: number,
  maxBars = 20
): Float64Array {
  const arr = ensureArray(close);
  const n = arr.length;
  const labels = new Float64Array(n);

  for (let i = 0; i < n - maxBars; i++) {
    const entry = arr[i];
    const tpLevel = entry * (1 + takeProfit);
    const slLevel = entry * (1 - stopLoss);

    for (let j = i + 1; j < Math.min(i + maxBars + 1, n); j++) {
      if (arr[j] >= tpLevel) {
        labels[i] = 1;
        break;
      } else if (arr[j] <= slLevel) {
        labels[i] = -1;
        break;
      }
    }
  }

  for (let i = Math.max(0, n - maxBars); i < n; i++) {
    labels[i] = NaN;
  }

  return labels;
}
