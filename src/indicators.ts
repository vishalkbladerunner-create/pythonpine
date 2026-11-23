/**
 * PythonPine Indicators - Convenience class grouping all technical indicators
 */

import { rsi, macd, stochastic, adx, cci, roc, tsi } from './momentum.js';
import { atr, bollingerBands, keltnerChannel, historicalVolatility } from './volatility.js';
import { sma, ema, wma, hma, supertrend, ichimoku } from './trend.js';
import { zscore, percentileRank, logReturns } from './normalize.js';

export class Indicators {
  // Momentum
  static rsi = rsi;
  static macd = macd;
  static stochastic = stochastic;
  static adx = adx;
  static cci = cci;
  static roc = roc;
  static tsi = tsi;

  // Volatility
  static atr = atr;
  static bollingerBands = bollingerBands;
  static keltnerChannel = keltnerChannel;
  static historicalVolatility = historicalVolatility;

  // Trend
  static sma = sma;
  static ema = ema;
  static wma = wma;
  static hma = hma;
  static supertrend = supertrend;
  static ichimoku = ichimoku;

  // Normalization
  static zscore = zscore;
  static percentileRank = percentileRank;
  static logReturns = logReturns;
}
