# SpatialForecast dependencies

QGIS provides the core PyQGIS runtime and normally includes NumPy, GDAL and Matplotlib.

For SES, Holt, Holt-Winters, ARIMA, SARIMA and AUTO, install `statsmodels` into the Python environment used by QGIS:

```bash
python -m pip install "statsmodels>=0.14"
```

SMA and WMA can run without statsmodels.

If QGIS uses an OSGeo4W environment on Windows, run the installation from the QGIS/OSGeo4W shell so the package is installed into the correct interpreter.
