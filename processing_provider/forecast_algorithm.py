from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from osgeo import gdal

logger = logging.getLogger("spatialforecast.forecast_algorithm")

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)

from .core_engine import (
    MODEL_CODES,
    MODEL_LABELS,
    MODEL_NAMES,
    STATSMODELS_AVAILABLE,
    aggregate_stack,
    fit_and_forecast,
    forecast_cube,
    prepare_series,
    refit_for_plot,
    representative_diagnostics,
)
from .preprocessing import align_inputs


class RasterTimeSeriesForecastAlgorithm(QgsProcessingAlgorithm):
    INPUT_RASTERS="INPUT_RASTERS"; REFERENCE_RASTER="REFERENCE_RASTER"; RESAMPLING="RESAMPLING"
    MODEL="MODEL"; FORECAST_STEPS="FORECAST_STEPS"; SEASONAL_PERIOD="SEASONAL_PERIOD"
    SMA_WINDOW="SMA_WINDOW"; WMA_WINDOW="WMA_WINDOW"; ARIMA_ORDER="ARIMA_ORDER"; SARIMA_ORDER="SARIMA_ORDER"
    HOLDOUT="HOLDOUT"; SPATIAL_MODE="SPATIAL_MODE"; AGGREGATION="AGGREGATION"; AGG_STAT="AGG_STAT"
    REP_PIXEL="REP_PIXEL"; START_DATE="START_DATE"; TIME_FREQUENCY="TIME_FREQUENCY"
    GRAPH_SET="GRAPH_SET"; GRAPH_FORMAT="GRAPH_FORMAT"; GRAPH_DPI="GRAPH_DPI"; OUTPUT_DIR="OUTPUT_DIR"

    MODELS=["SMA","WMA","SES","HOLT","HW_ADD","HW_MUL","ARIMA","SARIMA","AUTO"]
    MODEL_TEXT=["SMA — Simple moving average","WMA — Weighted moving average","SES — Simple exponential smoothing","Holt — linear trend","Holt-Winters additive","Holt-Winters multiplicative","ARIMA","SARIMA","AUTO — best model per pixel"]
    RESAMPLING_TEXT=["Bilinear (continuous rasters)","Nearest neighbour (categorical/discrete rasters)"]
    SPATIAL_MODES=["Native spatial resolution","Aggregate pixels for faster computation"]
    AGG_STATS=["Mean","Median"]
    FREQUENCIES=["Index only","Daily","Monthly","Quarterly","Annual"]
    GRAPH_SETS=["None","Essential publication figures","Full diagnostics"]
    GRAPH_FORMATS=["PNG","PDF","SVG"]

    def name(self): return "raster_time_series_forecast"
    def displayName(self): return "SpatialForecast — Spatial Time-Series Forecast"
    def group(self): return "Forecasting"
    def groupId(self): return "forecasting"
    def createInstance(self): return RasterTimeSeriesForecastAlgorithm()

    def shortHelpString(self):
        return (
            "Pixel-wise or block-wise raster time-series forecasting with eight curated methods: "
            "SMA, WMA, SES, Holt, Holt-Winters additive, Holt-Winters multiplicative, ARIMA and SARIMA. "
            "AUTO chooses the lowest holdout-RMSE model independently for each valid pixel. "
            "Inputs are automatically reprojected/cropped/resampled to an exactly aligned reference grid. "
            "By default the raster with the smallest spatial footprint is used as the reference; an explicit "
            "reference raster can be supplied. Produces forecast rasters, uncertainty rasters, model/diagnostic "
            "maps, aligned input rasters, CSV summaries and publication-quality graphs."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMultipleLayers(self.INPUT_RASTERS,"Ordered raster time series",layerType=QgsProcessing.SourceType.TypeRaster))
        self.addParameter(QgsProcessingParameterRasterLayer(self.REFERENCE_RASTER,"Optional reference raster (blank = automatically use smallest spatial extent)",optional=True))
        self.addParameter(QgsProcessingParameterEnum(self.RESAMPLING,"Resampling method for grid harmonisation",options=self.RESAMPLING_TEXT,defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.MODEL,"Forecast method",options=self.MODEL_TEXT,defaultValue=8))
        self.addParameter(QgsProcessingParameterNumber(self.FORECAST_STEPS,"Forecast horizon (time steps)",type=QgsProcessingParameterNumber.Type.Integer,minValue=1,defaultValue=12))
        self.addParameter(QgsProcessingParameterNumber(self.SEASONAL_PERIOD,"Seasonal period (e.g., 12 monthly, 4 quarterly)",type=QgsProcessingParameterNumber.Type.Integer,minValue=2,defaultValue=12))
        self.addParameter(QgsProcessingParameterNumber(self.SMA_WINDOW,"SMA window",type=QgsProcessingParameterNumber.Type.Integer,minValue=2,defaultValue=3))
        self.addParameter(QgsProcessingParameterNumber(self.WMA_WINDOW,"WMA window",type=QgsProcessingParameterNumber.Type.Integer,minValue=2,defaultValue=3))
        self.addParameter(QgsProcessingParameterString(self.ARIMA_ORDER,"ARIMA order p,d,q",defaultValue="1,1,1"))
        self.addParameter(QgsProcessingParameterString(self.SARIMA_ORDER,"SARIMA non-seasonal order p,d,q",defaultValue="1,1,1"))
        self.addParameter(QgsProcessingParameterNumber(self.HOLDOUT,"Temporal holdout observations for validation",type=QgsProcessingParameterNumber.Type.Integer,minValue=1,defaultValue=3))
        self.addParameter(QgsProcessingParameterEnum(self.SPATIAL_MODE,"Spatial processing mode",options=self.SPATIAL_MODES,defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.AGGREGATION,"Aggregation block size when aggregation is selected (2 = 2×2, 5 = 5×5)",type=QgsProcessingParameterNumber.Type.Integer,minValue=2,defaultValue=2))
        self.addParameter(QgsProcessingParameterEnum(self.AGG_STAT,"Aggregation statistic",options=self.AGG_STATS,defaultValue=0))
        self.addParameter(QgsProcessingParameterString(self.REP_PIXEL,"Representative pixel row,col for detailed graphs (blank = automatic)",optional=True,defaultValue=""))
        self.addParameter(QgsProcessingParameterString(self.START_DATE,"Optional first observation date (YYYY-MM-DD)",optional=True,defaultValue=""))
        self.addParameter(QgsProcessingParameterEnum(self.TIME_FREQUENCY,"Time frequency for graph axis",options=self.FREQUENCIES,defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.GRAPH_SET,"Publication graphs",options=self.GRAPH_SETS,defaultValue=2))
        self.addParameter(QgsProcessingParameterEnum(self.GRAPH_FORMAT,"Graph format",options=self.GRAPH_FORMATS,defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(self.GRAPH_DPI,"Graph DPI",type=QgsProcessingParameterNumber.Type.Integer,minValue=150,maxValue=1200,defaultValue=600))
        self.addParameter(QgsProcessingParameterFolderDestination(self.OUTPUT_DIR,"Output folder"))

    @staticmethod
    def _parse_order(text):
        try:
            x=tuple(int(v.strip()) for v in str(text).split(","))
            if len(x)!=3 or any(v<0 for v in x): raise ValueError
            return x
        except Exception:
            raise QgsProcessingException("Order must be p,d,q, for example 1,1,1.")

    @staticmethod
    def _layer_path(layer): return layer.source().split("|")[0]

    @staticmethod
    def _read_aligned_stack(paths):
        arrays=[]; gt=proj=None
        for p in paths:
            ds=gdal.Open(p)
            if ds is None: raise QgsProcessingException(f"Could not open aligned raster: {p}")
            if gt is None: gt=ds.GetGeoTransform(); proj=ds.GetProjection()
            for b in range(1,ds.RasterCount+1):
                band=ds.GetRasterBand(b); arr=band.ReadAsArray().astype(np.float64); nd=band.GetNoDataValue()
                if nd is not None: arr[np.isclose(arr,nd,equal_nan=True)]=np.nan
                arr[~np.isfinite(arr)]=np.nan; arrays.append(arr)
        if not arrays: raise QgsProcessingException("No raster bands found.")
        return np.stack(arrays,axis=0), tuple(gt), proj

    @staticmethod
    def _agg_gt(gt,block):
        x=list(gt); x[1]=gt[1]*block; x[2]=gt[2]*block; x[4]=gt[4]*block; x[5]=gt[5]*block; return tuple(x)

    @staticmethod
    def _write_raster(path,array,gt,proj,nodata=-9999.0,dtype=gdal.GDT_Float32):
        arr=np.asarray(array); rows,cols=arr.shape
        ds=gdal.GetDriverByName("GTiff").Create(str(path),cols,rows,1,dtype,options=["COMPRESS=DEFLATE","TILED=YES"])
        ds.SetGeoTransform(gt); ds.SetProjection(proj); band=ds.GetRasterBand(1)
        out=arr.copy()
        if np.issubdtype(out.dtype,np.floating): out[~np.isfinite(out)]=nodata
        band.WriteArray(out); band.SetNoDataValue(nodata); band.FlushCache(); ds.FlushCache(); ds=None

    @staticmethod
    def _rep_pixel(stack,requested):
        rows,cols=stack.shape[1:]
        if requested:
            try:
                r,c=[int(v.strip()) for v in requested.split(",")]
                if 0<=r<rows and 0<=c<cols and prepare_series(stack[:,r,c]) is not None: return r,c
            except Exception as exc:
                logger.debug("Requested representative pixel invalid (%s): %s", requested, exc)
        center=(rows//2,cols//2)
        candidates=[]
        for r in range(rows):
            for c in range(cols):
                if prepare_series(stack[:,r,c]) is not None:
                    d=(r-center[0])**2+(c-center[1])**2; candidates.append((d,r,c))
        if not candidates: raise QgsProcessingException("No valid pixel time series available.")
        _,r,c=min(candidates); return r,c

    @staticmethod
    def _time_axis(n_obs,n_future,start_date,freq):
        if not start_date or freq=="Index only": return np.arange(1,n_obs+1),np.arange(n_obs+1,n_obs+n_future+1),"Time step"
        try: cur=datetime.strptime(start_date,"%Y-%m-%d")
        except ValueError: raise QgsProcessingException("Start date must be YYYY-MM-DD.")
        vals=[]
        for _ in range(n_obs+n_future):
            vals.append(cur)
            if freq=="Daily": cur+=timedelta(days=1)
            elif freq=="Monthly":
                y=cur.year+(cur.month//12); m=(cur.month%12)+1; cur=cur.replace(year=y,month=m,day=min(cur.day,28))
            elif freq=="Quarterly":
                for __ in range(3):
                    y=cur.year+(cur.month//12); m=(cur.month%12)+1; cur=cur.replace(year=y,month=m,day=min(cur.day,28))
            elif freq=="Annual": cur=cur.replace(year=cur.year+1)
        return np.asarray(vals[:n_obs]),np.asarray(vals[n_obs:]),"Date"

    def _candidate_metrics_for_rep(self,series,steps,holdout,sma_window,wma_window,arima_order,sarima_order,seasonal_period):
        rows=[]
        for name in MODEL_CODES:
            try:
                fr=fit_and_forecast(series,name,steps,holdout,sma_window,wma_window,arima_order,sarima_order,seasonal_period)
                rows.append((name,fr.rmse,fr.mae,fr.smape,fr.aic,fr.bic,fr.ljungbox_p,fr.adf_p,fr.mean_interval_width))
            except Exception as exc:
                logger.warning("Model '%s' failed for representative pixel: %s", name, exc)
        return rows

    def _make_figures(self,out_dir,stack,result,rep_rc,requested_model,steps,holdout,sma_window,wma_window,arima_order,sarima_order,seasonal_period,graph_set,fmt,dpi,start_date,freq,feedback):
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            feedback.reportError(f"Graphs skipped: matplotlib unavailable: {exc}"); return []
        figdir=Path(out_dir)/"figures"; figdir.mkdir(exist_ok=True); ext=fmt.lower(); paths=[]
        plt.rcParams.update({"font.size":10,"axes.titlesize":12,"axes.labelsize":10,"legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,"savefig.dpi":dpi,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
        obs_x,fut_x,xlab=self._time_axis(stack.shape[0],steps,start_date,freq)
        r,c=rep_rc; code=int(result["model_code"][r,c]); selected=MODEL_NAMES.get(code, requested_model if requested_model!="AUTO" else "SMA")
        series=stack[:,r,c]

        try:
            y,fitted,fc,lo,hi,resid=refit_for_plot(series,selected,steps,sma_window,wma_window,arima_order,sarima_order,seasonal_period)
            ox=obs_x[-len(y):]
            fig,ax=plt.subplots(figsize=(7.4,4.6)); ax.plot(ox,y,marker="o",ms=3,lw=1.35,label="Observed"); ax.plot(ox,fitted,lw=1.15,label="Fitted"); ax.plot(fut_x,fc,marker="o",ms=3,lw=1.5,label="Forecast"); ax.fill_between(fut_x,lo,hi,alpha=.18,label="95% interval"); ax.set(title=f"Pixel forecast — {MODEL_LABELS.get(selected,selected)} (row {r}, col {c})",xlabel=xlab,ylabel="Raster value"); ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout(); p=figdir/f"01_pixel_forecast.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))
        except Exception as exc: feedback.reportError(f"Pixel forecast graph skipped: {exc}")

        with np.errstate(invalid="ignore"):
            obsmean=np.nanmean(stack.reshape(stack.shape[0],-1),axis=1); ff=result["forecast"].reshape(steps,-1); fm=np.nanmean(ff,axis=1); q10=np.nanpercentile(ff,10,axis=1); q90=np.nanpercentile(ff,90,axis=1)
        fig,ax=plt.subplots(figsize=(7.4,4.6)); ax.plot(obs_x,obsmean,marker="o",ms=3,lw=1.35,label="Observed spatial mean"); ax.plot(fut_x,fm,marker="o",ms=3,lw=1.5,label="Forecast spatial mean"); ax.fill_between(fut_x,q10,q90,alpha=.18,label="Spatial 10–90% range"); ax.set(title="Spatial mean forecast",xlabel=xlab,ylabel="Mean raster value"); ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout(); p=figdir/f"02_spatial_mean_forecast.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))

        metrics=self._candidate_metrics_for_rep(series,steps,holdout,sma_window,wma_window,arima_order,sarima_order,seasonal_period)
        if metrics:
            labels=[MODEL_LABELS.get(x[0],x[0]) for x in metrics]; x=np.arange(len(labels)); width=.25
            fig,ax=plt.subplots(figsize=(9.0,4.8)); ax.bar(x-width,[m[1] for m in metrics],width,label="RMSE"); ax.bar(x,[m[2] for m in metrics],width,label="MAE"); ax.bar(x+width,[m[3] for m in metrics],width,label="sMAPE (%)"); ax.set(title="Representative-pixel validation metrics",ylabel="Metric value"); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=30,ha="right"); ax.grid(axis="y",alpha=.2); ax.legend(frameon=False); fig.tight_layout(); p=figdir/f"03_model_validation_comparison.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))

        valid_codes=result["model_code"][result["model_code"]>0]
        if valid_codes.size:
            codes=sorted(np.unique(valid_codes)); labels=[MODEL_LABELS.get(MODEL_NAMES[int(k)],str(k)) for k in codes]; shares=[100*np.sum(valid_codes==k)/valid_codes.size for k in codes]
            fig,ax=plt.subplots(figsize=(7.8,4.5)); ax.bar(labels,shares); ax.set(title="Selected model share across valid pixels",ylabel="Share of pixels (%)"); ax.tick_params(axis="x",rotation=30); ax.grid(axis="y",alpha=.2); fig.tight_layout(); p=figdir/f"04_model_selection_share.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))

        if graph_set=="Full diagnostics":
            try:
                y,fitted,fc,lo,hi,resid=refit_for_plot(series,selected,steps,sma_window,wma_window,arima_order,sarima_order,seasonal_period); clean=resid[np.isfinite(resid)]
                fig,ax=plt.subplots(figsize=(7.4,4.2)); ax.plot(np.arange(1,len(resid)+1),resid,lw=1.1); ax.axhline(0,lw=1,ls="--"); ax.set(title="Representative-pixel residuals",xlabel="Observation",ylabel="Residual"); ax.grid(alpha=.2); fig.tight_layout(); p=figdir/f"05_residual_series.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))
                if clean.size:
                    fig,ax=plt.subplots(figsize=(6.4,4.2)); ax.hist(clean,bins=min(25,max(7,int(np.sqrt(clean.size))))); ax.set(title="Residual distribution",xlabel="Residual",ylabel="Frequency"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); p=figdir/f"06_residual_histogram.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))
            except Exception as exc: feedback.reportError(f"Residual graphs skipped: {exc}")

            dg=representative_diagnostics(series)
            if dg:
                a=dg["acf"]; fig,ax=plt.subplots(figsize=(7.0,4.2)); ax.stem(np.arange(len(a)),a); ax.axhline(0,lw=.8); ax.set(title="Autocorrelation function (ACF)",xlabel="Lag",ylabel="ACF"); ax.grid(alpha=.18); fig.tight_layout(); p=figdir/f"07_acf.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))
                pa=dg["pacf"]; fig,ax=plt.subplots(figsize=(7.0,4.2)); ax.stem(np.arange(len(pa)),pa); ax.axhline(0,lw=.8); ax.set(title="Partial autocorrelation function (PACF)",xlabel="Lag",ylabel="PACF"); ax.grid(alpha=.18); fig.tight_layout(); p=figdir/f"08_pacf.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))

            vals=[result[k][np.isfinite(result[k])] for k in ("rmse","mae","smape")]
            if any(v.size for v in vals):
                fig,ax=plt.subplots(figsize=(7.0,4.4)); usable=[v for v in vals if v.size]; labs=[lab for v,lab in zip(vals,["RMSE","MAE","sMAPE (%)"]) if v.size]; ax.boxplot(usable,labels=labs,showfliers=False); ax.set(title="Spatial distribution of validation errors",ylabel="Metric value"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); p=figdir/f"09_validation_error_boxplots.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))

            widths=result["upper"]-result["lower"]
            med=[]; p10=[]; p90=[]
            for i in range(steps):
                v=widths[i][np.isfinite(widths[i])]; med.append(float(np.median(v)) if v.size else np.nan); p10.append(float(np.percentile(v,10)) if v.size else np.nan); p90.append(float(np.percentile(v,90)) if v.size else np.nan)
            fig,ax=plt.subplots(figsize=(7.0,4.3)); h=np.arange(1,steps+1); ax.plot(h,med,marker="o",ms=3,label="Median interval width"); ax.fill_between(h,p10,p90,alpha=.18,label="Spatial 10–90% range"); ax.set(title="Forecast uncertainty by horizon",xlabel="Forecast horizon",ylabel="95% interval width"); ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout(); p=figdir/f"10_uncertainty_by_horizon.{ext}"; fig.savefig(p,dpi=dpi); plt.close(fig); paths.append(str(p))
        return paths,metrics

    def processAlgorithm(self,parameters,context,feedback):
        layers=self.parameterAsLayerList(parameters,self.INPUT_RASTERS,context)
        if not layers: raise QgsProcessingException("Select at least one raster.")
        paths=[self._layer_path(x) for x in layers]
        ref_layer=self.parameterAsRasterLayer(parameters,self.REFERENCE_RASTER,context); ref_path=self._layer_path(ref_layer) if ref_layer else None
        resampling="bilinear" if self.parameterAsEnum(parameters,self.RESAMPLING,context)==0 else "nearest"
        model=self.MODELS[self.parameterAsEnum(parameters,self.MODEL,context)]
        steps=self.parameterAsInt(parameters,self.FORECAST_STEPS,context); seasonal=self.parameterAsInt(parameters,self.SEASONAL_PERIOD,context); sw=self.parameterAsInt(parameters,self.SMA_WINDOW,context); ww=self.parameterAsInt(parameters,self.WMA_WINDOW,context); holdout=self.parameterAsInt(parameters,self.HOLDOUT,context)
        arima=self._parse_order(self.parameterAsString(parameters,self.ARIMA_ORDER,context)); sarima=self._parse_order(self.parameterAsString(parameters,self.SARIMA_ORDER,context))
        spatial_mode=self.SPATIAL_MODES[self.parameterAsEnum(parameters,self.SPATIAL_MODE,context)]; block=self.parameterAsInt(parameters,self.AGGREGATION,context); agg_stat=self.AGG_STATS[self.parameterAsEnum(parameters,self.AGG_STAT,context)].lower()
        rep=self.parameterAsString(parameters,self.REP_PIXEL,context).strip(); start=self.parameterAsString(parameters,self.START_DATE,context).strip(); freq=self.FREQUENCIES[self.parameterAsEnum(parameters,self.TIME_FREQUENCY,context)]
        graph_set=self.GRAPH_SETS[self.parameterAsEnum(parameters,self.GRAPH_SET,context)]; fmt=self.GRAPH_FORMATS[self.parameterAsEnum(parameters,self.GRAPH_FORMAT,context)]; dpi=self.parameterAsInt(parameters,self.GRAPH_DPI,context); out_dir=self.parameterAsString(parameters,self.OUTPUT_DIR,context); Path(out_dir).mkdir(parents=True,exist_ok=True)
        if model not in ("SMA","WMA") and not STATSMODELS_AVAILABLE: raise QgsProcessingException("statsmodels is required for SES/Holt/Holt-Winters/ARIMA/SARIMA/AUTO.")

        feedback.pushInfo("Harmonising raster grids…")
        try: reference,aligned=align_inputs(paths,out_dir,method=resampling,explicit_reference=ref_path)
        except Exception as exc: raise QgsProcessingException(f"Raster alignment failed: {exc}")
        feedback.pushInfo(f"Reference grid: {Path(reference.path).name}; {reference.width}×{reference.height} cells; resampling={resampling}.")
        feedback.pushInfo("Every aligned raster now has identical CRS, extent, dimensions and cell boundaries.")
        stack,gt,proj=self._read_aligned_stack(aligned)
        if stack.shape[0]<6: raise QgsProcessingException("At least 6 temporal observations are required; seasonal methods generally need much more.")
        if holdout>=stack.shape[0]-2: raise QgsProcessingException("Holdout is too large for the time-series length.")

        actual_block=1
        if spatial_mode.startswith("Aggregate"):
            actual_block=block; feedback.pushInfo(f"Aggregating aligned grid to {block}×{block} blocks using {agg_stat}…"); stack=aggregate_stack(stack,block,agg_stat); gt=self._agg_gt(gt,block)
        feedback.pushInfo(f"Forecast grid: {stack.shape[1]} rows × {stack.shape[2]} columns; time steps={stack.shape[0]}.")
        feedback.pushInfo(f"Running {model} forecasting…")
        result=forecast_cube(stack,model,steps,holdout,sw,ww,arima,sarima,seasonal,progress=feedback.setProgress,canceled=feedback.isCanceled)

        # forecast + intervals
        for i in range(steps):
            self._write_raster(Path(out_dir)/f"forecast_{i+1:03d}.tif",result["forecast"][i],gt,proj)
            self._write_raster(Path(out_dir)/f"forecast_lower95_{i+1:03d}.tif",result["lower"][i],gt,proj)
            self._write_raster(Path(out_dir)/f"forecast_upper95_{i+1:03d}.tif",result["upper"][i],gt,proj)
        diag_files={
            "validation_rmse.tif":"rmse","validation_mae.tif":"mae","validation_smape.tif":"smape","model_aic.tif":"aic","model_bic.tif":"bic",
            "ljungbox_pvalue.tif":"ljungbox_p","adf_pvalue.tif":"adf_p","forecast_uncertainty_width.tif":"uncertainty_width"
        }
        for fn,key in diag_files.items(): self._write_raster(Path(out_dir)/fn,result[key],gt,proj)
        self._write_raster(Path(out_dir)/"selected_model_code.tif",result["model_code"],gt,proj,nodata=0,dtype=gdal.GDT_Int16)

        # summary tables
        with open(Path(out_dir)/"model_code_key.csv","w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["code","short_name","model"]); [w.writerow([code,name,MODEL_LABELS[name]]) for name,code in MODEL_CODES.items()]
        with open(Path(out_dir)/"forecast_summary.csv","w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["step","mean","median","p10","p90","valid_pixels"])
            for i in range(steps):
                v=result["forecast"][i][np.isfinite(result["forecast"][i])]; w.writerow([i+1,float(np.mean(v)) if v.size else "",float(np.median(v)) if v.size else "",float(np.percentile(v,10)) if v.size else "",float(np.percentile(v,90)) if v.size else "",int(v.size)])

        figpaths=[]; metric_rows=[]
        if graph_set!="None":
            feedback.pushInfo(f"Creating {graph_set.lower()} at {dpi} DPI in {fmt} format…")
            rc=self._rep_pixel(stack,rep)
            figpaths,metric_rows=self._make_figures(out_dir,stack,result,rc,model,steps,holdout,sw,ww,arima,sarima,seasonal,graph_set,fmt,dpi,start,freq,feedback)
            with open(Path(out_dir)/"representative_pixel_model_statistics.csv","w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["model","RMSE","MAE","sMAPE_percent","AIC","BIC","LjungBox_p","ADF_p","mean_95_interval_width"]); w.writerows(metric_rows)

        with open(Path(out_dir)/"run_report.txt","w",encoding="utf-8") as f:
            f.write("SpatialForecast run report\n==========================\n")
            f.write("Concept and research design: Zainab Khan\n")
            f.write(f"Requested model: {model}\nReference grid: {reference.path}\nGrid harmonisation: {resampling}\n")
            f.write(f"Aligned input directory: {Path(out_dir)/'aligned_inputs'}\nSpatial mode: {spatial_mode}\nAggregation block: {actual_block}\nAggregation statistic: {agg_stat}\n")
            f.write(f"Temporal observations: {stack.shape[0]}\nForecast horizon: {steps}\nSeasonal period: {seasonal}\nHoldout: {holdout}\n")
            f.write("\nEight model family: SMA, WMA, SES, Holt, Holt-Winters additive, Holt-Winters multiplicative, ARIMA, SARIMA.\n")
            f.write("AUTO selects the lowest holdout-RMSE model per pixel.\n")
            f.write("Validation maps: RMSE, MAE, sMAPE. Diagnostics: AIC, BIC, ADF p-value, Ljung-Box p-value, forecast interval width.\n")
            f.write("ARIMA/SARIMA intervals are model-based. SMA/WMA/exponential-smoothing intervals are residual-based approximations.\n")
            f.write(f"Figures created: {len(figpaths)}\n")
        feedback.pushInfo(f"Finished. Results: {out_dir}")
        return {self.OUTPUT_DIR:out_dir}
