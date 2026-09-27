def classFactory(iface):
    from .plugin import SpatialForecastPlugin
    return SpatialForecastPlugin(iface)
