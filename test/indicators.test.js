import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ensureArray,
  rollingMean,
  rollingMax,
  rollingMin,
  rollingStd,
  ewmMean,
  wilderSmooth,
  trueRange,
  shift,
  diff,
  crossAbove,
  crossBelow,
  rsi,
  macd,
  stochastic,
  williamsR,
  cci,
  roc,
  momentum,
  adx,
  stochRsi,
  trix,
  ultimateOscillator,
  cmo,
  tsi,
  atr,
  bollingerBands,
  keltnerChannel,
  donchianChannel,
  bollingerBandwidth,
  bollingerPercentB,
  historicalVolatility,
  normalizedAtr,
  ulcerIndex,
  chaikinVolatility,
  massIndex,
  parkinsonVolatility,
  garmanKlassVolatility,
  sma,
  ema,
  wma,
  dema,
  tema,
  hma,
  vwma,
  kama,
  supertrend,
  aroon,
  ichimoku,
  linearRegression,
  parabolicSar,
  zeroLagEma,
  mcginleyDynamic,
  zscore,
  percentileRank,
  logReturns,
  simpleReturns,
  fractionalDiff,
  minmaxScale,
  robustScale,
  cusumFilter,
  tripleBarrierLabels,
  engulfing,
  doji,
  hammer,
  shootingStar,
  morningStar,
  eveningStar,
  insideBar,
  outsideBar,
  supportResistanceZones,
  higherHighsLowerLows,
  obv,
  vwap,
  adl,
  cmf,
  volumeOscillator,
  forceIndex,
  mfi,
  easeOfMovement,
  volumeRoc,
  volumeDelta,
  intradayIntensity,
  priceVolumeTrend,
  vwMacd,
  smoothedObv,
  klingerOscillator,
  volumeFlowIndicator,
  pvi,
  nvi,
  fisherTransform,
  hilbertTransform,
  htSine,
  htPhase,
  laguerreRsi,
  mamaFama,
  superSmoother,
  cyberCycle,
  zScore,
  rollingMeanStd,
  skewnessKurtosis,
  medianAbsoluteDeviation,
  Indicators
} from '../dist/index.js';

const close = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0, 109.0, 111.0, 113.0, 112.0, 115.0, 117.0, 116.0, 118.0, 120.0, 119.0];
const high = close.map(c => c + 1.0);
const low = close.map(c => c - 1.0);
const open = close.map(c => c - 0.5);
const volume = [1000, 1200, 1100, 1500, 1300, 1400, 1600, 1700, 1500, 1900, 1800, 2000, 2100, 1950, 2200, 2300, 2100, 2400, 2500, 2350];

test('PythonPine - Core Utils', () => {
  const arr = ensureArray([1, 2, 3, 4, 5]);
  assert.equal(arr.length, 5);
  assert.ok(arr instanceof Float64Array);

  const mean = rollingMean(arr, 3);
  assert.ok(Number.isNaN(mean[0]));
  assert.ok(Number.isNaN(mean[1]));
  assert.equal(mean[2], 2);
  assert.equal(mean[3], 3);
  assert.equal(mean[4], 4);

  const maxVal = rollingMax(arr, 3);
  assert.equal(maxVal[4], 5);

  const minVal = rollingMin(arr, 3);
  assert.equal(minVal[4], 3);

  const shifted = shift(arr, 1);
  assert.ok(Number.isNaN(shifted[0]));
  assert.equal(shifted[1], 1);

  const diffed = diff(arr, 1);
  assert.ok(Number.isNaN(diffed[0]));
  assert.equal(diffed[1], 1);

  const tr = trueRange([10, 12], [8, 9], [9, 11]);
  assert.equal(tr.length, 2);
  assert.equal(tr[0], 2); // 10 - 8

  const crossed = crossAbove([1, 5], [3, 4]);
  assert.equal(crossed[0], false);
  assert.equal(crossed[1], true);
});

test('PythonPine - Momentum Indicators', () => {
  const rsiVal = rsi(close, 5);
  assert.equal(rsiVal.length, close.length);
  assert.ok(Number.isNaN(rsiVal[0]));
  assert.ok(rsiVal[6] > 0 && rsiVal[6] < 100);

  const macdRes = macd(close, 3, 6, 3);
  assert.equal(macdRes.length, 3);
  assert.ok(macdRes.macd instanceof Float64Array);
  assert.ok(macdRes.signal instanceof Float64Array);
  assert.ok(macdRes.histogram instanceof Float64Array);

  const stoch = stochastic(close, high, low, 5, 3, 3);
  assert.ok(stoch.k instanceof Float64Array);
  assert.ok(stoch.d instanceof Float64Array);

  const wr = williamsR(close, high, low, 5);
  assert.equal(wr.length, close.length);

  const cciVal = cci(close, high, low, 5);
  assert.equal(cciVal.length, close.length);

  const rocVal = roc(close, 5);
  assert.equal(rocVal.length, close.length);

  const mom = momentum(close, 5);
  assert.equal(mom.length, close.length);

  const adxRes = adx(high, low, close, 5);
  assert.ok(adxRes.adx instanceof Float64Array);
  assert.ok(adxRes.plusDi instanceof Float64Array);
  assert.ok(adxRes.minusDi instanceof Float64Array);

  const stochRsiRes = stochRsi(close, 5, 5, 3, 3);
  assert.ok(stochRsiRes.k instanceof Float64Array);
  assert.ok(stochRsiRes.d instanceof Float64Array);

  const trixVal = trix(close, 3);
  assert.equal(trixVal.length, close.length);

  const uo = ultimateOscillator(high, low, close, 3, 5, 7);
  assert.equal(uo.length, close.length);

  const cmoVal = cmo(close, 5);
  assert.equal(cmoVal.length, close.length);

  const tsiVal = tsi(close, 5, 3);
  assert.equal(tsiVal.length, close.length);
});

test('PythonPine - Volatility Indicators', () => {
  const atrVal = atr(high, low, close, 5);
  assert.equal(atrVal.length, close.length);
  assert.ok(atrVal[6] > 0);

  const bb = bollingerBands(close, 5, 2.0);
  assert.ok(bb.upper instanceof Float64Array);
  assert.ok(bb.lower instanceof Float64Array);
  assert.ok(bb.middle instanceof Float64Array);
  assert.ok(bb.upper[6] > bb.middle[6]);
  assert.ok(bb.middle[6] > bb.lower[6]);

  const kc = keltnerChannel(high, low, close, 5, 2.0);
  assert.ok(kc.upper[6] > kc.middle[6]);

  const dc = donchianChannel(high, low, 5);
  assert.ok(dc.upper[6] >= dc.middle[6]);

  const bw = bollingerBandwidth(close, 5, 2.0);
  assert.equal(bw.length, close.length);

  const pctB = bollingerPercentB(close, 5, 2.0);
  assert.equal(pctB.length, close.length);

  const hv = historicalVolatility(close, 5);
  assert.equal(hv.length, close.length);

  const natr = normalizedAtr(high, low, close, 5);
  assert.equal(natr.length, close.length);

  const ui = ulcerIndex(close, 5);
  assert.equal(ui.length, close.length);

  const cv = chaikinVolatility(high, low, 5);
  assert.equal(cv.length, close.length);

  const mi = massIndex(high, low, 3, 5);
  assert.equal(mi.length, close.length);

  const pv = parkinsonVolatility(high, low, 5);
  assert.equal(pv.length, close.length);

  const gk = garmanKlassVolatility(open, high, low, close, 5);
  assert.equal(gk.length, close.length);
});

test('PythonPine - Trend Indicators', () => {
  const smaVal = sma(close, 5);
  assert.equal(smaVal[4], 102.2);

  const emaVal = ema(close, 5);
  assert.equal(emaVal[0], 100);

  const wmaVal = wma(close, 5);
  assert.equal(wmaVal.length, close.length);

  const demaVal = dema(close, 5);
  assert.equal(demaVal.length, close.length);

  const temaVal = tema(close, 5);
  assert.equal(temaVal.length, close.length);

  const hmaVal = hma(close, 5);
  assert.equal(hmaVal.length, close.length);

  const vwmaVal = vwma(close, volume, 5);
  assert.equal(vwmaVal.length, close.length);

  const kamaVal = kama(close, 5, 2, 10);
  assert.equal(kamaVal.length, close.length);

  const st = supertrend(high, low, close, 5, 3.0);
  assert.ok(st.trend instanceof Int32Array);
  assert.ok(st.supertrend instanceof Float64Array);

  const ar = aroon(high, low, 5);
  assert.ok(ar.aroonUp instanceof Float64Array);
  assert.ok(ar.aroonDown instanceof Float64Array);

  const ich = ichimoku(high, low, close, 3, 5, 7);
  assert.ok(ich.conversion instanceof Float64Array);
  assert.ok(ich.base instanceof Float64Array);

  const lr = linearRegression(close, 5);
  assert.equal(lr.length, close.length);

  const sar = parabolicSar(high, low);
  assert.equal(sar.length, close.length);

  const zlema = zeroLagEma(close, 5);
  assert.equal(zlema.length, close.length);

  const md = mcginleyDynamic(close, 5);
  assert.equal(md.length, close.length);
});

test('PythonPine - Normalization / ML Alpha', () => {
  const z = zscore(close, 5);
  assert.equal(z.length, close.length);

  const pr = percentileRank(close, 5);
  assert.equal(pr.length, close.length);

  const lr = logReturns(close);
  assert.ok(Number.isNaN(lr[0]));
  assert.ok(Math.abs(lr[1] - Math.log(102/100)) < 1e-10);

  const sr = simpleReturns(close);
  assert.ok(Number.isNaN(sr[0]));
  assert.ok(Math.abs(sr[1] - 0.02) < 1e-10);

  const fd = fractionalDiff(close, 0.4);
  assert.equal(fd.length, close.length);

  const mm = minmaxScale(close, 5);
  assert.ok(mm[5] >= 0 && mm[5] <= 1);

  const rs = robustScale(close, 5);
  assert.equal(rs.length, close.length);

  const cf = cusumFilter(close, 2.0);
  assert.equal(cf.length, close.length);

  const tb = tripleBarrierLabels(close, 0.02, 0.02, 5);
  assert.equal(tb.length, close.length);
});

test('PythonPine - Price Action', () => {
  const eng = engulfing([100, 98], [101, 103], [98, 97], [99, 102]);
  assert.equal(eng[1], 1); // Bullish engulfing

  const dj = doji([100], [105], [95], [100.1], 0.1);
  assert.equal(dj[0], 1);

  const hmr = hammer([100], [101], [90], [100.5]);
  assert.equal(hmr.length, 1);

  const ss = shootingStar([100], [110], [99], [99.5]);
  assert.equal(ss.length, 1);

  const ib = insideBar([110, 105], [90, 95]);
  assert.equal(ib[1], 1);

  const ob = outsideBar([105, 110], [95, 90]);
  assert.equal(ob[1], 1);

  const sr = supportResistanceZones(high, low, close, volume, 5, 2);
  assert.ok(sr.support instanceof Float64Array);
  assert.ok(sr.resistance instanceof Float64Array);

  const hll = higherHighsLowerLows(high, low, close, 3);
  assert.ok(hll.higherHigh instanceof Int32Array);
});

test('PythonPine - Volume Indicators', () => {
  const obvVal = obv(close, volume);
  assert.equal(obvVal.length, close.length);

  const vwapVal = vwap(high, low, close, volume);
  assert.equal(vwapVal.length, close.length);
  assert.ok(vwapVal[0] > 0);

  const adlVal = adl(high, low, close, volume);
  assert.equal(adlVal.length, close.length);

  const cmfVal = cmf(high, low, close, volume, 5);
  assert.equal(cmfVal.length, close.length);

  const vo = volumeOscillator(volume, 3, 5);
  assert.equal(vo.length, close.length);

  const fi = forceIndex(close, volume);
  assert.equal(fi.length, close.length);

  const mfiVal = mfi(high, low, close, volume, 5);
  assert.equal(mfiVal.length, close.length);

  const eom = easeOfMovement(high, low, volume, 5);
  assert.equal(eom.length, close.length);

  const vroc = volumeRoc(volume, 5);
  assert.equal(vroc.length, close.length);

  const vd = volumeDelta(volume, volume.map(v => v * 0.9));
  assert.equal(vd[0], 100);

  const ii = intradayIntensity(close, high, low, volume, 5);
  assert.equal(ii.length, close.length);

  const pvt = priceVolumeTrend(close, volume);
  assert.equal(pvt.length, close.length);

  const vwm = vwMacd(close, volume, 3, 6, 3);
  assert.ok(vwm.macd instanceof Float64Array);

  const sobv = smoothedObv(obvVal, 5);
  assert.equal(sobv.length, close.length);

  const kvo = klingerOscillator(high, low, close, volume, 5, 10, 3);
  assert.ok(kvo.kvo instanceof Float64Array);

  const vfi = volumeFlowIndicator(close, volume, 5);
  assert.equal(vfi.length, close.length);

  const pviVal = pvi(close, volume);
  assert.equal(pviVal.length, close.length);

  const nviVal = nvi(close, volume);
  assert.equal(nviVal.length, close.length);
});

test('PythonPine - Cycles & Statistical', () => {
  const fish = fisherTransform(close, 5);
  assert.equal(fish.length, close.length);

  const ht = hilbertTransform(close);
  assert.equal(ht.length, close.length);

  const sine = htSine(close);
  assert.ok(sine.sine instanceof Float64Array);

  const phase = htPhase(close);
  assert.equal(phase.length, close.length);

  const lrsi = laguerreRsi(close, 0.5);
  assert.equal(lrsi.length, close.length);

  const mf = mamaFama(close);
  assert.ok(mf.mama instanceof Float64Array);

  const ss = superSmoother(close, 5);
  assert.equal(ss.length, close.length);

  const cc = cyberCycle(close);
  assert.equal(cc.length, close.length);

  const zs = zScore(close, 5);
  assert.equal(zs.length, close.length);

  const rms = rollingMeanStd(close, 5);
  assert.ok(rms.means instanceof Float64Array);

  const sk = skewnessKurtosis(close, 5);
  assert.ok(sk.skewness instanceof Float64Array);

  const mad = medianAbsoluteDeviation(close, 5);
  assert.equal(mad.length, close.length);
});

test('PythonPine - Indicators Convenience Class', () => {
  const rsiVal = Indicators.rsi(close, 5);
  assert.equal(rsiVal.length, close.length);

  const macdVal = Indicators.macd(close, 3, 6, 3);
  assert.ok(macdVal.macd instanceof Float64Array);

  const smaVal = Indicators.sma(close, 5);
  assert.equal(smaVal[4], 102.2);

  const zVal = Indicators.zscore(close, 5);
  assert.equal(zVal.length, close.length);
});
