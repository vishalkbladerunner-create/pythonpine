/**
 * Reference-value and contract tests.
 *
 * Expected values were computed with independent Python implementations of the
 * Pine Script / TA-Lib definitions (SMA-seeded RMA). They pin down the numeric
 * behaviour that the smoke tests intentionally do not.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import {
  sma, ema, wma, rsi, atr, adx, macd, bollingerBands, stochastic, obv,
  supertrend, crossAbove, crossBelow, ensureArray, getWarmup,
} from '../dist/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
  45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64];
const high = close.map((c) => c + 1);
const low = close.map((c) => c - 1);
const volume = close.map((_, i) => 1000 + i * 10);

const closeTo = (actual, expected, eps = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= eps, `expected ${expected}, got ${actual}`);

test('moving averages match reference values', () => {
  closeTo(sma(close, 5)[4], 44.104);
  closeTo(sma(close, 5)[19], 46.06);
  closeTo(ema(close, 5)[19], 45.99609382307777);
  closeTo(wma(close, 5)[19], 46.02466666666667);

  // SMA has exactly length-1 leading NaNs.
  const s = sma(close, 5);
  assert.ok(s.slice(0, 4).every(Number.isNaN));
  assert.ok(!Number.isNaN(s[4]));
});

test('RSI matches Wilder/TA-Lib reference values', () => {
  const r = rsi(close, 14);
  assert.ok(r.slice(0, 14).every(Number.isNaN), 'first RSI value must be at index 14');
  closeTo(r[14], 70.46413502109705, 1e-9);
  closeTo(r[15], 66.24961855355505, 1e-9);
  closeTo(r[19], 57.91502067008556, 1e-9);
});

test('ATR is SMA-seeded and starts at length - 1', () => {
  const a = atr(high, low, close, 14);
  assert.ok(a.slice(0, 13).every(Number.isNaN));
  assert.ok(!Number.isNaN(a[13]));
  closeTo(a[13], 2.0, 1e-12);

  // Constant 2-point range: ATR must stay at 2 forever.
  closeTo(a[19], 2.0, 1e-12);
});

test('Bollinger Bands use population standard deviation', () => {
  const { upper, middle, lower } = bollingerBands(close, 5, 2);
  closeTo(middle[19], 46.06);
  closeTo(upper[19], 46.573030213535226, 1e-9);
  closeTo(lower[19], 45.54696978646478, 1e-9);

  for (let i = 19; i < close.length; i++) {
    assert.ok(upper[i] > middle[i] && middle[i] > lower[i]);
    closeTo((upper[i] - lower[i]) / 2, 2 * Math.sqrt(
      close.slice(i - 4, i + 1).reduce((s, v) => s + (v - middle[i]) ** 2, 0) / 5
    ), 1e-9);
  }
});

test('MACD tuple fields agree with reference values', () => {
  const [line, signal, hist] = macd(close, 3, 6, 3);
  closeTo(line[19], -0.06452204032783015, 1e-9);
  closeTo(signal[19], 0.017983848319382517, 1e-9);
  closeTo(hist[19], -0.08250588864721267, 1e-9);

  // Object access must mirror tuple access.
  const again = macd(close, 3, 6, 3);
  for (let i = 0; i < close.length; i++) assert.equal(again.macd[i], line[i]);
  // Histogram is exactly macd - signal.
  for (let i = 20; i < close.length; i++) closeTo(hist[i], line[i] - signal[i], 1e-12);
});

test('stochastic, ADX and supertrend stay in valid ranges', () => {
  const { k, d } = stochastic(close, high, low, 5, 3, 3);
  for (let i = 10; i < close.length; i++) {
    assert.ok(k[i] >= 0 && k[i] <= 100, `stochastic K out of range at ${i}: ${k[i]}`);
    assert.ok(d[i] >= 0 && d[i] <= 100);
  }

  const { adx: adxValues, plusDi, minusDi } = adx(high, low, close, 5);
  for (let i = 10; i < close.length; i++) {
    assert.ok(adxValues[i] >= 0 && adxValues[i] <= 100);
    assert.ok(plusDi[i] >= 0 && minusDi[i] >= 0);
  }

  const { trend, supertrend: line } = supertrend(high, low, close, 5, 3);
  for (let i = 5; i < close.length; i++) {
    assert.ok(trend[i] === 1 || trend[i] === -1);
    assert.ok(Number.isFinite(line[i]));
  }
});

test('OBV and cross helpers follow their definitions', () => {
  const o = obv(close, volume);
  assert.equal(o[0], 0);
  assert.equal(o[19], 4200);

  assert.deepEqual(
    Array.from(crossAbove([1, 5, 5, 2], [3, 4, 6, 3])),
    [false, true, false, false]
  );
  assert.deepEqual(
    Array.from(crossBelow([1, 5, 5, 2], [3, 4, 6, 3])),
    [false, false, true, false]
  );
});

test('warm-up helper agrees with actual first valid bar', () => {
  assert.equal(getWarmup('rsi', 14), 14);
  assert.equal(getWarmup('atr', 14), 13);
  assert.equal(getWarmup('sma', 20), 19);
  assert.equal(rsi(close, 14).findIndex((v) => !Number.isNaN(v)), getWarmup('rsi', 14));
  assert.equal(atr(high, low, close, 14).findIndex((v) => !Number.isNaN(v)), getWarmup('atr', 14));
});

test('empty, short and over-long windows degrade to NaN, never throw', () => {
  assert.equal(sma([], 14).length, 0);
  assert.equal(rsi([], 14).length, 0);
  assert.equal(macd([])[0].length, 0);

  assert.ok(sma([1, 2, 3], 14).every(Number.isNaN));
  assert.ok(rsi([1, 2, 3], 14).every(Number.isNaN));

  const a = ensureArray([1, 2, 3]);
  assert.ok(a instanceof Float64Array);
  a[0] = 99;
  assert.equal(ensureArray([1, 2, 3])[0], 1, 'ensureArray must copy');
});

test('package loads through CommonJS', () => {
  const out = execFileSync(process.execPath, [
    '-e',
    "const p=require('./dist/index.cjs');process.stdout.write(String(p.sma([1,2,3,4,5],3)[4]))",
  ], { cwd: join(here, '..'), encoding: 'utf8' });
  assert.equal(out, '4');
});
