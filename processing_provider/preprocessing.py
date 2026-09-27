from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from osgeo import gdal, osr

logger = logging.getLogger("spatialforecast.preprocessing")


@dataclass
class RasterGrid:
    path: str
    width: int
    height: int
    geotransform: tuple
    projection: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def cells(self):
        return self.width * self.height

    @property
    def bbox_area_native(self):
        return abs((self.xmax - self.xmin) * (self.ymax - self.ymin))


def inspect_raster(path: str) -> RasterGrid:
    ds = gdal.Open(path)
    if ds is None:
        raise ValueError(f"Could not open raster: {path}")
    gt = ds.GetGeoTransform()
    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
        raise ValueError("Rotated/skewed rasters are not supported as reference grids in this version.")
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + ds.RasterXSize * gt[1]
    ymin = ymax + ds.RasterYSize * gt[5]
    return RasterGrid(path, ds.RasterXSize, ds.RasterYSize, tuple(gt), ds.GetProjection(), min(xmin,xmax), min(ymin,ymax), max(xmin,xmax), max(ymin,ymax))


def _bbox_in_projection(grid: RasterGrid, target_wkt: str):
    if not grid.projection or not target_wkt or grid.projection == target_wkt:
        return grid.xmin, grid.ymin, grid.xmax, grid.ymax
    src = osr.SpatialReference(); src.ImportFromWkt(grid.projection)
    dst = osr.SpatialReference(); dst.ImportFromWkt(target_wkt)
    try:
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except Exception as exc:
        logger.debug("Axis mapping strategy not applied: %s", exc)
    tx = osr.CoordinateTransformation(src, dst)
    pts = [tx.TransformPoint(x,y)[:2] for x,y in [(grid.xmin,grid.ymin),(grid.xmin,grid.ymax),(grid.xmax,grid.ymin),(grid.xmax,grid.ymax)]]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return min(xs),min(ys),max(xs),max(ys)


def choose_smallest_extent(paths: List[str]) -> RasterGrid:
    grids = [inspect_raster(p) for p in paths]
    target_wkt = grids[0].projection
    scored = []
    for g in grids:
        xmin,ymin,xmax,ymax = _bbox_in_projection(g, target_wkt)
        area = abs((xmax-xmin)*(ymax-ymin))
        scored.append((area, g.cells, g))
    return min(scored, key=lambda x:(x[0],x[1]))[2]


def warp_to_reference(src_path: str, dst_path: str, reference: RasterGrid, method="bilinear", dst_nodata=-9999.0):
    alg = "near" if method.lower().startswith("near") else "bilinear"
    opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS=reference.projection or None,
        outputBounds=(reference.xmin, reference.ymin, reference.xmax, reference.ymax),
        width=reference.width,
        height=reference.height,
        resampleAlg=alg,
        dstNodata=dst_nodata,
        multithread=True,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    out = gdal.Warp(dst_path, src_path, options=opts)
    if out is None:
        raise RuntimeError(f"GDAL warp failed for {src_path}")
    out.FlushCache(); out = None
    return dst_path


def align_inputs(paths: List[str], out_dir: str, method="bilinear", explicit_reference: Optional[str]=None):
    ref = inspect_raster(explicit_reference) if explicit_reference else choose_smallest_extent(paths)
    aligned_dir = Path(out_dir) / "aligned_inputs"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    outputs=[]
    for i,p in enumerate(paths, start=1):
        dst = aligned_dir / f"aligned_{i:03d}_{Path(p).stem}.tif"
        warp_to_reference(p, str(dst), ref, method=method)
        outputs.append(str(dst))
    return ref, outputs
