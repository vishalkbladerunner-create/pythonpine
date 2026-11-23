# pythonpine

**Vectorized technical-analysis indicators for JavaScript and TypeScript.** 90+ Pine Script-compatible indicators over `Float64Array`, with zero runtime dependencies.

```bash
npm install pythonpine
```

```ts
import { rsi, macd, bollingerBands, supertrend } from 'pythonpine';

const close = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10];
const rsiValues = rsi(close, 14);          // Float64Array
const [macdLine, signal, hist] = macd(close);
const { upper, middle, lower } = bollingerBands(close, 20, 2);
```

Works in ESM, CommonJS, and TypeScript:

```js
// CommonJS
const { rsi, atr } = require('pythonpine');
```

## Why this library

- **Typed arrays in, typed arrays out.** Every indicator returns a `Float64Array` (or a tuple/object of them), allocated once per call. No boxing, no objects per bar.
- **Batch/vectorized API.** You pass the whole series and get the whole series back, which is what backtests and ML feature pipelines actually need.
- **Pine Script conventions.** Wilder smoothing (RMA) is SMA-seeded, Bollinger Bands use population standard deviation, and seeded warm-up bars are `NaN` instead of silently wrong numbers.
- **Zero runtime dependencies.** `esbuild` and `typescript` are dev-only.
- **Dual module output.** ESM (`dist/index.js`) with full `.d.ts` types, plus a bundled CJS build (`dist/index.cjs`).

## Semantics you should know

| Topic | Behaviour |
| --- | --- |
| Warm-up | Bars that cannot be computed yet are `NaN`. Check with `Number.isNaN(out[i])`. |
| Input types | `number[]`, `Float64Array`, or `Float32Array`. Inputs are copied, never mutated. |
| Output | `Float64Array` for series; tuples that also carry named properties, e.g. `const [k, d] = stochastic(...)` and `res.k`. |
| Empty input | Returns an empty `Float64Array`; no exceptions. |
| Short input | Returns all-`NaN` rather than throwing. |
| Seeding | EMA starts at the first observation (`pandas ewm(adjust=False)`). RMA is seeded with the SMA of the first `length` values (Pine / TA-Lib). |
| Timeframes | Indicators are timeframe-agnostic: pass bars of any interval. |

## Indicators

### Momentum

| Function | Signature |
| --- | --- |
| `rsi` | `(source, length = 14)` |
| `macd` | `(source, fast = 12, slow = 26, signal = 9)` → `[macd, signal, histogram]` |
| `stochastic` | `(close, high, low, length = 14, smoothK = 3, smoothD = 3)` → `[k, d]` |
| `stochRsi` | `(source, rsiLength = 14, stochLength = 14, smoothK = 3, smoothD = 3)` |
| `williamsR` | `(close, high, low, length = 14)` |
| `cci` | `(close, high, low, length = 20)` |
| `roc` | `(source, length = 12)` |
| `momentum` | `(source, length = 10)` |
| `adx` | `(high, low, close, length = 14)` → `[adx, plusDi, minusDi]` |
| `trix` | `(source, length = 15)` |
| `ultimateOscillator` | `(high, low, close, s1 = 7, s2 = 14, s3 = 28)` |
| `cmo` | `(source, length = 14)` |
| `tsi` | `(source, longLength = 25, shortLength = 13)` |

### Volatility

| Function | Signature |
| --- | --- |
| `atr` | `(high, low, close, length = 14)` |
| `bollingerBands` | `(source, length = 20, mult = 2)` → `{ upper, middle, lower }` |
| `bollingerBandwidth` | `(source, length = 20, mult = 2)` |
| `bollingerPercentB` | `(source, length = 20, mult = 2)` |
| `keltnerChannel` | `(high, low, close, length = 20, mult = 2)` |
| `donchianChannel` | `(high, low, length = 20)` |
| `historicalVolatility` | `(source, length = 20, annualize = true)` |
| `normalizedAtr` | `(high, low, close, length = 14)` |
| `ulcerIndex` | `(source, length = 14)` |
| `chaikinVolatility` | `(high, low, length = 10)` |
| `massIndex` | `(high, low, emaLength = 9, sumLength = 25)` |
| `parkinsonVolatility` | `(high, low, length = 10)` |
| `garmanKlassVolatility` | `(open, high, low, close, length = 10)` |

### Trend and moving averages

| Function | Signature |
| --- | --- |
| `sma` / `ema` / `wma` | `(source, length = 14)` |
| `dema` / `tema` | `(source, length = 14)` |
| `hma` | `(source, length = 14)` |
| `vwma` | `(source, volume, length = 14)` |
| `kama` | `(source, length = 10, fast = 2, slow = 30)` |
| `supertrend` | `(high, low, close, length = 10, mult = 3)` → `[trend, line]` |
| `aroon` | `(high, low, length = 14)` → `[aroonUp, aroonDown]` |
| `ichimoku` | `(high, low, close, tenkan = 9, kijun = 26, senkou = 52)` |
| `linearRegression` | `(source, length = 14)` |
| `parabolicSar` | `(high, low, afStep = 0.02, afMax = 0.2)` |
| `zeroLagEma` | `(source, length = 14)` |
| `mcginleyDynamic` | `(source, length = 14)` |

### Price action

`engulfing`, `doji`, `hammer`, `shootingStar`, `morningStar`, `eveningStar`, `insideBar`, `outsideBar`, `supportResistanceZones`, `higherHighsLowerLows`.

Pattern functions take `(open, high, low, close)` and return a boolean/`Int32Array` mask aligned to the input bars.

### Volume

`obv`, `vwap`, `adl`, `cmf`, `volumeOscillator`, `forceIndex`, `mfi`, `easeOfMovement`, `volumeRoc`, `volumeDelta`, `intradayIntensity`, `priceVolumeTrend`, `vwMacd`, `smoothedObv`, `klingerOscillator`, `volumeFlowIndicator`, `pvi`, `nvi`.

Note: `vwap` is cumulative from the start of the supplied series. For a session-anchored VWAP, pass only the bars of that session.

### Normalization and ML features

`zscore`, `percentileRank`, `logReturns`, `simpleReturns`, `fractionalDiff`, `minmaxScale`, `robustScale`, `cusumFilter`, `tripleBarrierLabels`.

### Cycles and statistics

`fisherTransform`, `hilbertTransform`, `htSine`, `htPhase`, `laguerreRsi`, `mamaFama`, `superSmoother`, `cyberCycle`, `zScore`, `rollingMeanStd`, `skewnessKurtosis`, `medianAbsoluteDeviation`.

### Utilities

`rollingMean`, `rollingStd`, `rollingMax`, `rollingMin`, `rollingSum`, `ewmMean`, `ewmAlpha`, `wilderSmooth`, `trueRange`, `diff`, `shift`, `crossAbove`, `crossBelow`, `highest`, `lowest`, `getWarmup`.

## Examples

### Signal detection

```ts
import { ema, crossAbove, crossBelow } from 'pythonpine';

const fast = ema(close, 12);
const slow = ema(close, 26);
const goldenCross = crossAbove(fast, slow);   // boolean[]
const deathCross = crossBelow(fast, slow);
```

### Vectorised features for a model

```ts
import { rsi, atr, zscore, logReturns } from 'pythonpine';

const features = close.map((_, i) => [
  rsi(close, 14)[i],
  atr(high, low, close, 14)[i],
  zscore(close, 20)[i],
  logReturns(close)[i],
]);
```

Compute each indicator once and index into it; the calls above are shown inline for brevity.

### Supertrend

```ts
import { supertrend } from 'pythonpine';

const { trend, supertrend: line } = supertrend(high, low, close, 10, 3);
// trend[i] === 1 -> uptrend, -1 -> downtrend
```

## TypeScript

Types ship with the package; no `@types` package needed.

```ts
import type { MacdResult, SupertrendResult } from 'pythonpine';
```

The default export is a convenience class whose members are the same functions:

```ts
import { Indicators } from 'pythonpine';

Indicators.rsi(close, 14);
Indicators.bollingerBands(close, 20, 2);
```

## Performance

Indicators are single-pass or windowed loops over `Float64Array`, with no intermediate allocations beyond the result. On an Apple M1, 100k bars take roughly 1–2 ms for RSI(14) and EMA(20), and 3–4 ms for ADX(14) or Bollinger Bands. Rolling windows are O(n·window); for very long windows on millions of bars, pre-aggregate or subsample.

## Development

```bash
npm install
npm run build     # ESM + CJS + types
npm test          # node:test suite
```

## License

MIT
