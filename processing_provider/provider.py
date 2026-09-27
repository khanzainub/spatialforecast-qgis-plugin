from qgis.core import QgsProcessingProvider
from .forecast_algorithm import RasterTimeSeriesForecastAlgorithm


class SpatialForecastProvider(QgsProcessingProvider):
    def id(self):
        return "spatialforecast"

    def name(self):
        return "SpatialForecast"

    def longName(self):
        return "SpatialForecast — Raster Time-Series Forecasting"

    def loadAlgorithms(self):
        self.addAlgorithm(RasterTimeSeriesForecastAlgorithm())
