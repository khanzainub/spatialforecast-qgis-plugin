# SpatialForecast 1.0 — QGIS spatial time-series forecasting

**Concept and research design: Zainab Khan**

SpatialForecast is a QGIS Processing plugin for pixel-wise and block-wise raster time-series forecasting. It is designed for environmental and geospatial time series where the same variable is observed repeatedly over space.

## Eight curated forecasting methods

1. **SMA — Simple Moving Average**: transparent baseline/smoothing model.
2. **WMA — Weighted Moving Average**: recent observations receive greater weight.
3. **SES — Simple Exponential Smoothing**: level-only series without systematic trend/seasonality.
4. **Holt Linear Trend**: series with a persistent trend but no seasonal cycle.
5. **Holt-Winters Additive**: trend + roughly constant seasonal amplitude.
6. **Holt-Winters Multiplicative**: trend + seasonality whose amplitude scales with the level; requires strictly positive values.
7. **ARIMA**: non-seasonal autocorrelated time series.
8. **SARIMA**: autocorrelation plus a seasonal cycle.

**AUTO** is not a ninth model. It evaluates the eight available model families that can be fitted to a pixel and selects the one with the lowest temporal holdout RMSE.

## Raster harmonisation before forecasting

Input rasters do **not** need to have identical extents, cell sizes or CRS.

By default SpatialForecast:

1. identifies the input raster with the **smallest spatial footprint**;
2. uses that raster as the reference grid;
3. reprojects other rasters to the reference CRS when necessary;
4. crops/warps them to the reference extent;
5. forces exactly the same output rows, columns and cell boundaries;
6. resamples using either:
   - **Bilinear** — recommended for continuous variables such as temperature, rainfall, NDVI, LST, soil moisture; or
   - **Nearest neighbour** — recommended when values are discrete/categorical and must not be interpolated.

An optional explicit reference raster can override automatic reference selection.

All harmonised inputs are retained in `aligned_inputs/` so the preprocessing can be visually inspected.

> Version 0.2 supports ordinary north-up reference rasters. Rotated/skewed reference grids are rejected rather than silently misaligning cells.

## Spatial computation mode

Choose either:

- **Native spatial resolution** — one time-series model for every valid original reference-grid pixel; or
- **Aggregate pixels for faster computation** — 2×2, 3×3, 5×5, etc. blocks, using mean or median aggregation.

The latter is valuable for large rasters because ARIMA/SARIMA and AUTO model fitting can be computationally expensive.

## Forecast raster outputs

For every requested future time step:

- `forecast_001.tif`, `forecast_002.tif`, ...
- `forecast_lower95_001.tif`, ...
- `forecast_upper95_001.tif`, ...

## Statistical / diagnostic raster outputs

- `validation_rmse.tif` — holdout root mean squared error
- `validation_mae.tif` — holdout mean absolute error
- `validation_smape.tif` — symmetric mean absolute percentage error
- `model_aic.tif` — Akaike information criterion when available
- `model_bic.tif` — Bayesian information criterion when available
- `adf_pvalue.tif` — Augmented Dickey-Fuller stationarity-test p-value
- `ljungbox_pvalue.tif` — residual autocorrelation diagnostic p-value
- `forecast_uncertainty_width.tif` — mean width of the 95% forecast interval
- `selected_model_code.tif` — best/selected model spatial map

AIC/BIC are model-family diagnostics and may be NoData for methods where they are not naturally provided.

## Publication-quality figures

Choose **None**, **Essential**, or **Full diagnostics**.

Essential figures:

1. observed + fitted + forecast time series for a representative pixel, with 95% uncertainty band;
2. spatial-mean observed + forecast series, with spatial 10–90% envelope;
3. representative-pixel comparison of candidate model validation metrics;
4. spatial share of selected models.

Full diagnostics additionally creates:

5. residual time-series plot;
6. residual histogram;
7. ACF plot;
8. PACF plot;
9. spatial validation-error boxplots;
10. forecast-uncertainty versus forecast horizon plot.

Formats:

- PNG, default **600 DPI**
- PDF
- SVG

The tool also exports the underlying statistics to CSV so figures can be reproduced or restyled for manuscripts.

## Important interpretation notes

- Holdout RMSE/MAE/sMAPE evaluate forecast performance on the most recent observations withheld from fitting.
- Lower RMSE/MAE/sMAPE generally indicates better holdout accuracy for the same pixel and target variable.
- ADF and Ljung-Box p-values should be interpreted as diagnostic evidence, not as model rankings by themselves.
- ARIMA/SARIMA uncertainty intervals come from the fitted statistical model.
- SMA, WMA, SES, Holt and Holt-Winters use residual-based approximate 95% forecast intervals in v0.2.
- AUTO chooses by holdout RMSE, not by AIC/BIC.

## Installation

1. Unzip `spatialforecast_qgis_plugin_v1.0.zip`.
2. Copy the `spatialforecast` directory into the active QGIS profile's `python/plugins` folder.
3. Restart QGIS.
4. Enable **SpatialForecast** from **Plugins → Manage and Install Plugins**.
5. Open **Processing Toolbox → SpatialForecast → Forecasting**.

## Dependencies

QGIS normally supplies NumPy, GDAL and Matplotlib.

SES/Holt/Holt-Winters/ARIMA/SARIMA/AUTO require `statsmodels` in the same Python environment used by QGIS:

```bash
python -m pip install statsmodels
```

SMA and WMA do not require statsmodels.
