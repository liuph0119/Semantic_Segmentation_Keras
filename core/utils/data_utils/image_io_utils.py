"""
utilization for load/save images, note that all the image are loaded to a [3-dim] array.
"""
import cv2
from PIL import Image
import os
import osr
from osgeo import gdal
import numpy as np
from keras.preprocessing.image import load_img, img_to_array


def _load_image_gdal(image_path, value_scale=1.0):
    """ using gdal to read image, especially for remote sensing multi-spectral images
    :param image_path: string, image path
    :param value_scale: float, default 1.0. the data array will divided by the 'value_scale'

    :return: array of shape (height, width, band)
    """
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    col=ds.RasterXSize
    row=ds.RasterYSize
    band=ds.RasterCount

    img=np.zeros((row, col, band))
    for i in range(band):
        dt = ds.GetRasterBand(i+1)
        img[:,:,i] = dt.ReadAsArray(0, 0, col, row)

    return img / value_scale


def load_image(image_path, is_gray=False, value_scale=1, target_size=None, use_gdal=False):
    """ load image
    :param image_path: string, image path
    :param is_gray: bool, default False
    :param value_scale: float, default 1. the data array will divided by the 'value_scale'
    :param target_size: tuple, default None. target spatial size to resize. If None, no resize will be performed.
    :param use_gdal: bool, default False.  whether use gdal to load data,  this is usually used for loading
    multi-spectral images, or images with geo-spatial information

    :return: array of shape (height, width, band)
    """
    assert value_scale!=0
    if use_gdal:
        # if use gdal, resize and gray are valid
        return _load_image_gdal(image_path, value_scale) / value_scale
    if is_gray:
        try:
            img = img_to_array(load_img(image_path, color_mode="grayscale", target_size=target_size))
        except:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if target_size is not None:
                img = cv2.resize(img, target_size)
                img = np.expand_dims(img, axis=-1)
    else:
        try:
            img = img_to_array(load_img(image_path, target_size=target_size))
        except:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            if target_size is not None:
                img = cv2.resize(img, target_size)

    return img / value_scale


def get_image_info(image_path, get_rows=False, get_cols=False, get_bands=False, get_geotransform=False, get_projection=False, get_nodatavalue=False):
    """ get the basic information of a image
    :param image_path: string
        image path
    :param get_rows: bool, default False
        whether to get rows
    :param get_cols: bool, default False
        whether to get cols
    :param get_bands: bool, default False
        whether to get bands
    :param get_geotransform: bool, default False
        whether to get geo-transform
    :param get_projection: bool, default False
        whether to get spatial reference system
    :param get_nodatavalue: bool, default False
        whether to get no-data value

    :return: dict
    """
    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    tif_info = dict()

    if get_rows:
        tif_info["rows"] = ds.RasterYSize
    if get_cols:
        tif_info["cols"] = ds.RasterXSize
    if get_bands:
        tif_info["bands"] = ds.RasterCount
    if get_geotransform:
        tif_info["geotransform"] = ds.GetGeoTransform()
    if get_projection:
        tif_info["projection"] = ds.GetProjection()
    if get_nodatavalue:
        tif_info["nodatavalue"] = ds.GetRasterBand(1).GetNoDataValue()

    del ds
    return tif_info


def save_to_image(arr, image_path):
    """ save common-formatted images
    :param arr: array of shape (height, width, 3) or (height, width)
    :param image_path: string

    :return: None
    """
    Image.fromarray(arr.astype(np.uint8)).save(image_path)


def save_to_image_gdal(arr, image_path, datatype=gdal.GDT_Byte, geoTransform = (0,1,0,0,0,-1), proj=None, nodata=None):
    """ save to geo-tiff image
    :param arr: array of shape (height, width, 3) or (height, width)
    :param image_path: string
    :param datatype: data type of the geo-tiff image, default gdal.GDT_Byte.
    :param geoTransform: tuple, default (0,1,0,0,0,-1)
        the geo-transform parameters.
    :param proj: string, default None
        the spatial projection wkt string. If None, no spatial projection will added to the output image
    :param nodata: float, default None
        no-data value. If None, no no-data value will be set

    :return: None
    """
    datatype = gdal.GDT_Byte if datatype is None else datatype

    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    if arr.ndim != 3:
        raise ValueError("[save_to_image_gdal: the input array must have 2 or 3 dimensions!!!]")

    nBands = arr.shape[-1]
    nRows, nCols = arr.shape[:2]

    driver = gdal.GetDriverByName("GTiff")
    if os.path.exists(image_path):
        os.remove(image_path)

    outRaster = driver.Create(image_path, nCols, nRows, nBands, datatype)
    outRaster.SetGeoTransform(geoTransform)
    if proj is not None:
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromWkt(proj)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())

    for _bandNum in range(nBands):
        outBand = outRaster.GetRasterBand(_bandNum+1)
        if nodata is not None:
            outBand.SetNoDataValue(nodata)
        outBand.WriteArray(arr[:,:, _bandNum])
        outBand.FlushCache()