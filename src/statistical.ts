/**
 * PythonPine Statistical - Statistical and distribution indicators
 */

import { ArrayLike, ensureArray } from './utils.js';

export type RollingMeanStdResult = [Float64Array, Float64Array] & {
  means: Float64Array;
  stds: Float64Array;
};

export type SkewnessKurtosisResult = [Float64Array, Float64Array] & {
  skewness: Float64Array;
  kurtosis: Float64Array;
};

/**
 * Z-score of Price (0.0 filled warmup for legacy compatibility)
 */
export function zScore(priceList: ArrayLike, period = 20): Float64Array {
  const p = ensureArray(priceList);
  const n = p.length;
  const z = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < period - 1) {
      z[i] = 0.0;
    } else {
      let sum = 0;
      for (let j = i - period + 1; j <= i; j++) {
        sum += p[j];
      }
      const mean = sum / period;

      let varSum = 0;
      for (let j = i - period + 1; j <= i; j++) {
        const diff = p[j] - mean;
        varSum += diff * diff;
      }
      const std = Math.sqrt(varSum / period);
      z[i] = std !== 0 ? (p[i] - mean) / std : 0.0;
    }
  }

  return z;
}

/**
 * Rolling Mean and Std
 */
export function rollingMeanStd(
  priceList: ArrayLike,
  period = 20
): RollingMeanStdResult {
  const p = ensureArray(priceList);
  const n = p.length;
  const means = new Float64Array(n);
  const stds = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < period - 1) {
      means[i] = 0.0;
      stds[i] = 0.0;
    } else {
      let sum = 0;
      for (let j = i - period + 1; j <= i; j++) {
        sum += p[j];
      }
      const mean = sum / period;

      let varSum = 0;
      for (let j = i - period + 1; j <= i; j++) {
        const diff = p[j] - mean;
        varSum += diff * diff;
      }
      means[i] = mean;
      stds[i] = Math.sqrt(varSum / period);
    }
  }

  const result = [means, stds] as RollingMeanStdResult;
  result.means = means;
  result.stds = stds;
  return result;
}

/**
 * Skewness and Kurtosis
 */
export function skewnessKurtosis(
  priceList: ArrayLike,
  period = 20
): SkewnessKurtosisResult {
  const p = ensureArray(priceList);
  const n = p.length;
  const skewVals = new Float64Array(n);
  const kurtVals = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < period - 1) {
      skewVals[i] = 0.0;
      kurtVals[i] = 0.0;
    } else {
      let sum = 0;
      for (let j = i - period + 1; j <= i; j++) {
        sum += p[j];
      }
      const mean = sum / period;

      let m2 = 0;
      let m3 = 0;
      let m4 = 0;
      for (let j = i - period + 1; j <= i; j++) {
        const diff = p[j] - mean;
        const diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
        m4 += diff2 * diff2;
      }
      m2 /= period;
      m3 /= period;
      m4 /= period;

      const std = Math.sqrt(m2);
      skewVals[i] = std !== 0 ? m3 / Math.pow(std, 3) : 0.0;
      kurtVals[i] = m2 !== 0 ? m4 / (m2 * m2) - 3.0 : 0.0; // Excess kurtosis
    }
  }

  const result = [skewVals, kurtVals] as SkewnessKurtosisResult;
  result.skewness = skewVals;
  result.kurtosis = kurtVals;
  return result;
}

/**
 * Median Absolute Deviation (MAD)
 */
export function medianAbsoluteDeviation(
  priceList: ArrayLike,
  period = 20
): Float64Array {
  const p = ensureArray(priceList);
  const n = p.length;
  const mad = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    if (i < period - 1) {
      mad[i] = 0.0;
    } else {
      const windowVals: number[] = [];
      for (let j = i - period + 1; j <= i; j++) {
        windowVals.push(p[j]);
      }
      windowVals.sort((a, b) => a - b);
      const median = windowVals[Math.floor(period / 2)];

      const devVals: number[] = [];
      for (let j = 0; j < period; j++) {
        devVals.push(Math.abs(windowVals[j] - median));
      }
      devVals.sort((a, b) => a - b);
      mad[i] = devVals[Math.floor(period / 2)];
    }
  }

  return mad;
}
