/**
 * PythonPine Utils - Core vectorized math helpers
 * High-performance array and rolling operations for technical indicators.
 */

export type ArrayLike = number[] | Float64Array | Float32Array;

/**
 * Convert input to Float64Array.
 */
export function ensureArray(data: ArrayLike): Float64Array {
  if (data instanceof Float64Array) {
    return new Float64Array(data);
  }
  return new Float64Array(data);
}

/**
 * Replace first `warmup` values with NaN for stability.
 */
export function padNan(result: Float64Array, warmup: number): Float64Array {
  if (warmup > 0 && result.length >= warmup) {
    for (let i = 0; i < warmup; i++) {
      result[i] = NaN;
    }
  }
  return result;
}

/**
 * Vectorized rolling maximum over `window`.
 * Matches pandas rolling(window, min_periods=1).max()
 */
export function rollingMax(arr: ArrayLike, window: number): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - window + 1);
    let maxVal = -Infinity;
    for (let j = start; j <= i; j++) {
      const v = data[j];
      if (!Number.isNaN(v) && v > maxVal) {
        maxVal = v;
      }
    }
    result[i] = maxVal === -Infinity ? NaN : maxVal;
  }
  return result;
}

/**
 * Vectorized rolling minimum over `window`.
 * Matches pandas rolling(window, min_periods=1).min()
 */
export function rollingMin(arr: ArrayLike, window: number): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - window + 1);
    let minVal = Infinity;
    for (let j = start; j <= i; j++) {
      const v = data[j];
      if (!Number.isNaN(v) && v < minVal) {
        minVal = v;
      }
    }
    result[i] = minVal === Infinity ? NaN : minVal;
  }
  return result;
}

/**
 * Vectorized rolling sum over `window`.
 * Matches pandas rolling(window, min_periods=1).sum()
 */
export function rollingSum(arr: ArrayLike, window: number, minPeriods = 1): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  let currentSum = 0;
  let count = 0;

  for (let i = 0; i < n; i++) {
    const v = data[i];
    if (!Number.isNaN(v)) {
      currentSum += v;
      count++;
    }

    if (i >= window) {
      const oldVal = data[i - window];
      if (!Number.isNaN(oldVal)) {
        currentSum -= oldVal;
        count--;
      }
    }

    result[i] = (count >= minPeriods) ? currentSum : NaN;
  }
  return result;
}

/**
 * Vectorized rolling mean (SMA) over `window`.
 * Matches pandas rolling(window, min_periods=window).mean()
 */
export function rollingMean(arr: ArrayLike, window: number, minPeriods = window): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  let sum = 0;
  let validCount = 0;

  for (let i = 0; i < n; i++) {
    const v = data[i];
    if (!Number.isNaN(v)) {
      sum += v;
      validCount++;
    }

    if (i >= window) {
      const prev = data[i - window];
      if (!Number.isNaN(prev)) {
        sum -= prev;
        validCount--;
      }
    }

    if (i + 1 >= minPeriods && validCount >= minPeriods) {
      result[i] = sum / validCount;
    } else {
      result[i] = NaN;
    }
  }
  return result;
}

/**
 * Vectorized rolling standard deviation.
 * Matches pandas rolling(window, min_periods=window).std(ddof=ddof)
 */
export function rollingStd(arr: ArrayLike, window: number, ddof = 0): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  for (let i = 0; i < n; i++) {
    if (i < window - 1) {
      result[i] = NaN;
      continue;
    }

    let sum = 0;
    let count = 0;
    for (let j = i - window + 1; j <= i; j++) {
      const v = data[j];
      if (!Number.isNaN(v)) {
        sum += v;
        count++;
      }
    }

    if (count <= ddof) {
      result[i] = NaN;
      continue;
    }

    const mean = sum / count;
    let varianceSum = 0;
    for (let j = i - window + 1; j <= i; j++) {
      const v = data[j];
      if (!Number.isNaN(v)) {
        const d = v - mean;
        varianceSum += d * d;
      }
    }

    result[i] = Math.sqrt(varianceSum / (count - ddof));
  }
  return result;
}

/**
 * Exponential Weighted Moving Average (EWM) with span.
 * Matches pandas ewm(span=span, adjust=adjust).mean()
 */
export function ewmMean(arr: ArrayLike, span: number, adjust = false): Float64Array {
  const alpha = 2 / (span + 1);
  return ewmAlpha(arr, alpha, adjust);
}

/**
 * EWM with explicit alpha.
 * Matches pandas ewm(alpha=alpha, adjust=adjust).mean()
 */
export function ewmAlpha(arr: ArrayLike, alpha: number, adjust = false): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  let firstValidIdx = -1;
  for (let i = 0; i < n; i++) {
    if (!Number.isNaN(data[i])) {
      firstValidIdx = i;
      break;
    }
    result[i] = NaN;
  }

  if (firstValidIdx === -1) {
    return result;
  }

  if (!adjust) {
    // Recursive definition: y[i] = (1 - alpha) * y[i-1] + alpha * x[i]
    result[firstValidIdx] = data[firstValidIdx];
    for (let i = firstValidIdx + 1; i < n; i++) {
      const x = data[i];
      if (Number.isNaN(x)) {
        result[i] = result[i - 1];
      } else {
        result[i] = (1 - alpha) * result[i - 1] + alpha * x;
      }
    }
  } else {
    // Adjusted definition
    let weightedSum = 0;
    let sumOfWeights = 0;
    for (let i = 0; i < n; i++) {
      if (i < firstValidIdx) continue;
      const x = data[i];
      if (!Number.isNaN(x)) {
        weightedSum = (1 - alpha) * weightedSum + x;
        sumOfWeights = (1 - alpha) * sumOfWeights + 1;
        result[i] = weightedSum / sumOfWeights;
      } else {
        result[i] = result[i - 1];
      }
    }
  }

  return result;
}

/**
 * Wilder's smoothing method (RMA) - used in RSI, ATR, ADX.
 * The series is seeded with the SMA of the first `period` valid values
 * starting at `start` (Pine Script / TA-Lib convention), then recursed as:
 *   rma[i] = (rma[i-1] * (period - 1) + arr[i]) / period
 * Values before the seed are NaN.
 *
 * @param start Index of the first observation used for the seed. Use 1 for
 * series whose first element is a placeholder (e.g. price changes, DMs).
 */
export function wilderSmooth(arr: ArrayLike, period: number, start = 0): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n).fill(NaN);
  if (n === 0 || period < 1) return result;

  const seedIdx = start + period - 1;
  if (seedIdx >= n) return result;

  let sum = 0;
  let count = 0;
  for (let i = start; i <= seedIdx; i++) {
    if (!Number.isNaN(data[i])) {
      sum += data[i];
      count++;
    }
  }
  if (count === 0) return result;

  result[seedIdx] = sum / count;
  for (let i = seedIdx + 1; i < n; i++) {
    const x = data[i];
    result[i] = Number.isNaN(x)
      ? result[i - 1]
      : (result[i - 1] * (period - 1) + x) / period;
  }
  return result;
}

/**
 * Vectorized True Range calculation.
 */
export function trueRange(high: ArrayLike, low: ArrayLike, close: ArrayLike): Float64Array {
  const h = ensureArray(high);
  const l = ensureArray(low);
  const c = ensureArray(close);
  const n = c.length;
  const tr = new Float64Array(n);
  if (n === 0) return tr;

  tr[0] = h[0] - l[0];
  for (let i = 1; i < n; i++) {
    const prevC = c[i - 1];
    const tr1 = h[i] - l[i];
    const tr2 = Math.abs(h[i] - prevC);
    const tr3 = Math.abs(l[i] - prevC);
    tr[i] = Math.max(tr1, tr2, tr3);
  }
  return tr;
}

/**
 * Vectorized difference: arr[i] - arr[i - periods]
 */
export function diff(arr: ArrayLike, periods = 1): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (i < periods) {
      result[i] = NaN;
    } else {
      result[i] = data[i] - data[i - periods];
    }
  }
  return result;
}

/**
 * Shift array by n periods.
 */
export function shift(arr: ArrayLike, periods = 1): Float64Array {
  const data = ensureArray(arr);
  const n = data.length;
  const result = new Float64Array(n);

  if (periods > 0) {
    for (let i = 0; i < periods && i < n; i++) {
      result[i] = NaN;
    }
    for (let i = periods; i < n; i++) {
      result[i] = data[i - periods];
    }
  } else if (periods < 0) {
    const absP = Math.abs(periods);
    for (let i = 0; i < n - absP; i++) {
      result[i] = data[i + absP];
    }
    for (let i = Math.max(0, n - absP); i < n; i++) {
      result[i] = NaN;
    }
  } else {
    result.set(data);
  }
  return result;
}

/**
 * Detect when arr1 crosses above arr2.
 */
export function crossAbove(arr1: ArrayLike, arr2: ArrayLike): boolean[] {
  const a1 = ensureArray(arr1);
  const a2 = ensureArray(arr2);
  const n = a1.length;
  const result = new Array<boolean>(n).fill(false);

  for (let i = 1; i < n; i++) {
    result[i] = a1[i] > a2[i] && a1[i - 1] <= a2[i - 1];
  }
  return result;
}

/**
 * Detect when arr1 crosses below arr2.
 */
export function crossBelow(arr1: ArrayLike, arr2: ArrayLike): boolean[] {
  const a1 = ensureArray(arr1);
  const a2 = ensureArray(arr2);
  const n = a1.length;
  const result = new Array<boolean>(n).fill(false);

  for (let i = 1; i < n; i++) {
    result[i] = a1[i] < a2[i] && a1[i - 1] >= a2[i - 1];
  }
  return result;
}

/**
 * Highest value over last `length` bars.
 */
export function highest(arr: ArrayLike, length: number): Float64Array {
  return rollingMax(arr, length);
}

/**
 * Lowest value over last `length` bars.
 */
export function lowest(arr: ArrayLike, length: number): Float64Array {
  return rollingMin(arr, length);
}

// Index of the first non-NaN value produced by each indicator.
export const WARMUP_PERIODS: Record<string, (...args: number[]) => number> = {
  sma: (p = 14) => p - 1,
  ema: () => 0,
  rsi: (p = 14) => p,
  atr: (p = 14) => p - 1,
  macd: (fast = 12, slow = 26, signal = 9) => slow + signal,
  bollinger: (p = 20) => p - 1,
  adx: (p = 14) => p * 2,
};

/**
 * Index of the first non-NaN output value for a known indicator, so callers
 * can slice off the warm-up region without scanning for NaN.
 */
export function getWarmup(indicator: string, ...args: number[]): number {
  if (indicator in WARMUP_PERIODS) {
    return WARMUP_PERIODS[indicator](...args);
  }
  return 14;
}
