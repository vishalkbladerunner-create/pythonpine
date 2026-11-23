/**
 * PythonPine Volume - Volume-Based Indicators
 */

import {
  ArrayLike,
  ensureArray,
  ewmMean
} from './utils.js';

export type VwMacdResult = [Float64Array, Float64Array, Float64Array] & {
  macd: Float64Array;
  signal: Float64Array;
  histogram: Float64Array;
};

export type KlingerResult = [Float64Array, Float64Array] & {
  kvo: Float64Array;
  signal: Float64Array;
};

/**
 * On Balance Volume (OBV)
 */
export function obv(closeList: ArrayLike, volumeList: ArrayLike): Float64Array {
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;
  const result = new Float64Array(n);

  let currentObv = 0;
  for (let i = 1; i < n; i++) {
    if (c[i] > c[i - 1]) {
      currentObv += v[i];
    } else if (c[i] < c[i - 1]) {
      currentObv -= v[i];
    }
    result[i] = currentObv;
  }
  return result;
}

/**
 * Volume Weighted Average Price (VWAP)
 */
export function vwap(
  highList: ArrayLike,
  lowList: ArrayLike,
  closeList: ArrayLike,
  volumeList: ArrayLike
): Float64Array {
  const h = ensureArray(highList);
  const l = ensureArray(lowList);
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;

  const result = new Float64Array(n);
  let cumVol = 0;
  let cumVp = 0;

  for (let i = 0; i < n; i++) {
    const typical = (h[i] + l[i] + c[i]) / 3;
    cumVol += v[i];
    cumVp += typical * v[i];
    result[i] = cumVol !== 0 ? cumVp / cumVol : 0;
  }
  return result;
}

/**
 * Accumulation/Distribution Line (ADL)
 */
export function adl(
  highList: ArrayLike,
  lowList: ArrayLike,
  closeList: ArrayLike,
  volumeList: ArrayLike
): Float64Array {
  const h = ensureArray(highList);
  const l = ensureArray(lowList);
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;

  const result = new Float64Array(n);
  let runningSum = 0;

  for (let i = 0; i < n; i++) {
    const rangeVal = h[i] - l[i] === 0 ? 1e-10 : h[i] - l[i];
    const mfm = ((c[i] - l[i]) - (h[i] - c[i])) / rangeVal;
    const mfv = mfm * v[i];
    runningSum += mfv;
    result[i] = runningSum;
  }
  return result;
}

/**
 * Chaikin Money Flow (CMF)
 */
export function cmf(
  highList: ArrayLike,
  lowList: ArrayLike,
  closeList: ArrayLike,
  volumeList: ArrayLike,
  period = 20
): Float64Array {
  const h = ensureArray(highList);
  const l = ensureArray(lowList);
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;

  const mfv = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const rangeVal = h[i] - l[i] === 0 ? 1e-10 : h[i] - l[i];
    const mfm = ((c[i] - l[i]) - (h[i] - c[i])) / rangeVal;
    mfv[i] = mfm * v[i];
  }

  const result = new Float64Array(n);
  for (let i = period - 1; i < n; i++) {
    let sumMfv = 0;
    let sumVol = 0;
    for (let j = i - period + 1; j <= i; j++) {
      sumMfv += mfv[j];
      sumVol += v[j];
    }
    result[i] = sumMfv / (sumVol === 0 ? 1e-10 : sumVol);
  }
  return result;
}

/**
 * Volume Oscillator
 */
export function volumeOscillator(
  volumeList: ArrayLike,
  shortPeriod = 14,
  longPeriod = 28
): Float64Array {
  const v = ensureArray(volumeList);
  const n = v.length;
  const result = new Float64Array(n);

  for (let i = longPeriod - 1; i < n; i++) {
    let shortSum = 0;
    for (let j = i - shortPeriod + 1; j <= i; j++) {
      shortSum += v[j];
    }
    const shortMa = shortSum / shortPeriod;

    let longSum = 0;
    for (let j = i - longPeriod + 1; j <= i; j++) {
      longSum += v[j];
    }
    const longMa = longSum / longPeriod;

    result[i] = longMa !== 0 ? ((shortMa - longMa) / longMa) * 100 : 0;
  }
  return result;
}

/**
 * Force Index
 */
export function forceIndex(
  closeList: ArrayLike,
  volumeList: ArrayLike
): Float64Array {
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;
  const result = new Float64Array(n);

  for (let i = 1; i < n; i++) {
    result[i] = (c[i] - c[i - 1]) * v[i];
  }
  return result;
}

/**
 * Money Flow Index (MFI)
 */
export function mfi(
  highList: ArrayLike,
  lowList: ArrayLike,
  closeList: ArrayLike,
  volumeList: ArrayLike,
  period = 14
): Float64Array {
  const h = ensureArray(highList);
  const l = ensureArray(lowList);
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;

  const tp = new Float64Array(n);
  const rawMf = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    tp[i] = (h[i] + l[i] + c[i]) / 3;
    rawMf[i] = tp[i] * v[i];
  }

  const posFlow = new Float64Array(n);
  const negFlow = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    if (tp[i] > tp[i - 1]) posFlow[i] = rawMf[i];
    else if (tp[i] < tp[i - 1]) negFlow[i] = rawMf[i];
  }

  const result = new Float64Array(n);
  for (let i = period; i < n; i++) {
    let posSum = 0;
    let negSum = 0;
    for (let j = i - period + 1; j <= i; j++) {
      posSum += posFlow[j];
      negSum += negFlow[j];
    }

    if (negSum === 0) {
      result[i] = 100;
    } else {
      const mr = posSum / negSum;
      result[i] = 100 - (100 / (1 + mr));
    }
  }
  return result;
}

/**
 * Ease of Movement (EOM)
 */
export function easeOfMovement(
  highList: ArrayLike,
  lowList: ArrayLike,
  volumeList: ArrayLike,
  period = 14
): Float64Array {
  const h = ensureArray(highList);
  const l = ensureArray(lowList);
  const v = ensureArray(volumeList);
  const n = h.length;

  const emv = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    const dm = (h[i] + l[i]) / 2 - (h[i - 1] + l[i - 1]) / 2;
    const br = v[i] / (h[i] - l[i] === 0 ? 1e-10 : h[i] - l[i]);
    emv[i] = br === 0 ? 0 : dm / br;
  }

  const result = new Float64Array(n);
  for (let i = period - 1; i < n; i++) {
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) {
      sum += emv[j];
    }
    result[i] = sum / period;
  }
  return result;
}

/**
 * Volume Rate of Change (VROC)
 */
export function volumeRoc(volumeList: ArrayLike, period = 14): Float64Array {
  const v = ensureArray(volumeList);
  const n = v.length;
  const result = new Float64Array(n);

  for (let i = period; i < n; i++) {
    const prev = v[i - period];
    result[i] = prev !== 0 ? ((v[i] - prev) / prev) * 100 : 0;
  }
  return result;
}

/**
 * Volume Delta (buy/sell imbalance)
 */
export function volumeDelta(
  buyVolume: ArrayLike,
  sellVolume: ArrayLike
): Float64Array {
  const b = ensureArray(buyVolume);
  const s = ensureArray(sellVolume);
  const n = b.length;
  const result = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    result[i] = b[i] - s[i];
  }
  return result;
}

/**
 * Intraday Intensity
 */
export function intradayIntensity(
  closeList: ArrayLike,
  highList: ArrayLike,
  lowList: ArrayLike,
  volumeList: ArrayLike,
  period = 14
): Float64Array {
  const c = ensureArray(closeList);
  const h = ensureArray(highList);
  const l = ensureArray(lowList);
  const v = ensureArray(volumeList);
  const n = c.length;

  const ii = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const rangeVal = h[i] - l[i] === 0 ? 1e-10 : h[i] - l[i];
    ii[i] = ((2 * c[i] - h[i] - l[i]) / rangeVal) * v[i];
  }

  const result = new Float64Array(n);
  for (let i = period - 1; i < n; i++) {
    let sumIi = 0;
    let sumV = 0;
    for (let j = i - period + 1; j <= i; j++) {
      sumIi += ii[j];
      sumV += v[j];
    }
    result[i] = sumIi / (sumV === 0 ? 1e-10 : sumV);
  }
  return result;
}

/**
 * Price Volume Trend (PVT)
 */
export function priceVolumeTrend(
  closeList: ArrayLike,
  volumeList: ArrayLike
): Float64Array {
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;
  const result = new Float64Array(n);

  let currentPvt = 0;
  for (let i = 1; i < n; i++) {
    const change = c[i - 1] !== 0 ? (c[i] - c[i - 1]) / c[i - 1] : 0;
    currentPvt += change * v[i];
    result[i] = currentPvt;
  }
  return result;
}

/**
 * Volume-Weighted MACD
 */
export function vwMacd(
  closeList: ArrayLike,
  volumeList: ArrayLike,
  shortPeriod = 12,
  longPeriod = 26,
  signalPeriod = 9
): VwMacdResult {
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;

  const weightedPrice = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const volSafe = v[i] === 0 ? 1e-10 : v[i];
    weightedPrice[i] = (c[i] * v[i]) / volSafe;
  }

  const shortEma = ewmMean(weightedPrice, shortPeriod);
  const longEma = ewmMean(weightedPrice, longPeriod);
  const macdLine = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    macdLine[i] = shortEma[i] - longEma[i];
  }

  const signalLine = ewmMean(macdLine, signalPeriod);
  const histogram = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    histogram[i] = macdLine[i] - signalLine[i];
  }

  const result = [macdLine, signalLine, histogram] as VwMacdResult;
  result.macd = macdLine;
  result.signal = signalLine;
  result.histogram = histogram;
  return result;
}

/**
 * Smoothed OBV
 */
export function smoothedObv(
  obvList: ArrayLike,
  period = 14
): Float64Array {
  return ewmMean(obvList, period);
}

/**
 * Klinger Volume Oscillator
 */
export function klingerOscillator(
  highList: ArrayLike,
  lowList: ArrayLike,
  closeList: ArrayLike,
  volumeList: ArrayLike,
  fast = 34,
  slow = 55,
  signal = 13
): KlingerResult {
  const h = ensureArray(highList);
  const l = ensureArray(lowList);
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = h.length;

  const dm = new Float64Array(n);
  const trend = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    dm[i] = h[i] - l[i];
    if (i > 0) {
      const tp = (h[i] + l[i] + c[i]) / 3;
      const prevTp = (h[i - 1] + l[i - 1] + c[i - 1]) / 3;
      trend[i] = tp > prevTp ? 1 : -1;
    }
  }

  const vf = new Float64Array(n);
  let cm = 0;

  for (let i = 1; i < n; i++) {
    cm += Math.abs(dm[i] - dm[i - 1]);
    const ratio = cm !== 0 ? dm[i] / cm : 0;
    vf[i] = v[i] * trend[i] * 100 * ratio;
  }

  const fastEma = ewmMean(vf, fast);
  const slowEma = ewmMean(vf, slow);
  const kvo = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    kvo[i] = fastEma[i] - slowEma[i];
  }

  const signalLine = ewmMean(kvo, signal);

  const result = [kvo, signalLine] as KlingerResult;
  result.kvo = kvo;
  result.signal = signalLine;
  return result;
}

/**
 * Volume Flow Indicator (VFI)
 */
export function volumeFlowIndicator(
  closeList: ArrayLike,
  volumeList: ArrayLike,
  period = 130,
  coef = 0.2
): Float64Array {
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;

  const vfi = new Float64Array(n);

  for (let i = 1; i < n; i++) {
    let volSum = 0;
    let count = 0;
    for (let j = Math.max(0, i - period + 1); j <= i; j++) {
      volSum += v[j];
      count++;
    }
    const avgVol = volSum / (count || 1);
    const logRet = c[i - 1] !== 0 ? Math.log(c[i] / c[i - 1]) : 0;
    const cutOff = coef * avgVol;
    const effectiveV = v[i] < 2 * cutOff ? v[i] : cutOff;
    vfi[i] = logRet > 0 ? effectiveV * logRet : 0;
  }

  return ewmMean(vfi, period);
}

/**
 * Positive Volume Index (PVI)
 */
export function pvi(
  closeList: ArrayLike,
  volumeList: ArrayLike
): Float64Array {
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  result[0] = 1000.0;
  for (let i = 1; i < n; i++) {
    if (v[i] > v[i - 1]) {
      const change = c[i - 1] !== 0 ? (c[i] - c[i - 1]) / c[i - 1] : 0;
      result[i] = result[i - 1] + result[i - 1] * change;
    } else {
      result[i] = result[i - 1];
    }
  }
  return result;
}

/**
 * Negative Volume Index (NVI)
 */
export function nvi(
  closeList: ArrayLike,
  volumeList: ArrayLike
): Float64Array {
  const c = ensureArray(closeList);
  const v = ensureArray(volumeList);
  const n = c.length;
  const result = new Float64Array(n);
  if (n === 0) return result;

  result[0] = 1000.0;
  for (let i = 1; i < n; i++) {
    if (v[i] < v[i - 1]) {
      const change = c[i - 1] !== 0 ? (c[i] - c[i - 1]) / c[i - 1] : 0;
      result[i] = result[i - 1] + result[i - 1] * change;
    } else {
      result[i] = result[i - 1];
    }
  }
  return result;
}
