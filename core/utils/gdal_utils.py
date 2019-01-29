"""
    Script: gdal_utils.py
    Author: Penghua Liu
    Date: 2019-01-13
    Email: liuphhhh@foxmail.com
    Functions: some util functions for geo-spatial image process

"""

from osgeo import gdal
import os
import osr
import numpy as np


def getGeoInfomation(fn):
    """ Get the geo-transofrm information and the projection information.

    # Args:
        :param fn: geo image file name, string

    # Returns
        geoTransform: the geotransform tuple of 6 numbers, corresponding to
                      [top_left_x, pixel_width, north rotation(0), top left y, north rotation(0), -pixel_height]
        proj: the spatial projection, can be exported to wkt
    """

    img = gdal.Open(fn, gdal.GA_ReadOnly)
    geoTransform, proj =  img.GetGeoTransform(), img.GetProjection()
    del img
    return geoTransform, proj


def arr_to_tif(arr, fn, datatype=None, geoTransform = (0,1,0,0,0,-1), proj="", nodata=None):
    """ save a given array to a geo-tiff file

    # Args:
        :param arr: a 2-d array or a 3-d array
        :param fn: the save file name, string
        :param datatype: raster data type, None for default(thus be gdal.Byte)
        :param geoTransform: the geotransform papameter tuple
        :param proj: the spatial projection
        :param nodata: no data value

    # Returns:
        None
    """
    if datatype is None:
        datatype = gdal.GDT_Byte

    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    if arr.ndim != 3:
        return

    nBands = arr.shape[-1]
    nRows, nCols = arr.shape[:2]

    driver = gdal.GetDriverByName("GTiff")
    if os.path.exists(fn):
        os.remove(fn)

    outRaster = driver.Create(fn, nCols, nRows, nBands, datatype)
    outRaster.SetGeoTransform(geoTransform)
    if proj != "":
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromWkt(proj)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())

    for _bandNum in range(nBands):
        outBand = outRaster.GetRasterBand(_bandNum+1)
        if nodata is not None:
            outBand.SetNoDataValue(nodata)
        outBand.WriteArray(arr[:,:, _bandNum])
        outBand.FlushCache()