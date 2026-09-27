# Changelog

## 0.2.0
- Expanded forecasting core to 8 curated methods: SMA, WMA, SES, Holt, Holt-Winters additive, Holt-Winters multiplicative, ARIMA, SARIMA.
- AUTO now selects among the eight fit-capable methods per pixel using holdout RMSE.
- Added automatic raster harmonisation to the smallest spatial-footprint raster or an explicit user reference.
- Added selectable bilinear / nearest-neighbour resampling.
- Aligned inputs are written to disk for inspection.
- Added native-resolution versus block-aggregation modes, with mean/median aggregation.
- Added pixel-wise RMSE, MAE, sMAPE, AIC, BIC, ADF p-value, Ljung-Box p-value and uncertainty-width outputs.
- Added 600-DPI PNG, PDF and SVG publication figure options.
- Added observed/fitted/forecast, spatial mean, model comparison, model share, residual, ACF, PACF, validation distribution and uncertainty-horizon figures.
