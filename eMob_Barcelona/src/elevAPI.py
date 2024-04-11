from osgeo import gdal, osr
from os import listdir
from os.path import isfile, join
import json
from rtree import index
import os.path
from os import path
import numpy as np
import math

import os

import matplotlib
import matplotlib.pyplot as plt


# Simplified version of https://github.com/Jorl17/open-elevation, without server interfaces
class GDALInterface(object):
    SEA_LEVEL = 0

    def __init__(self, tif_path):
        super(GDALInterface, self).__init__()
        self.tif_path = tif_path
        self.loadMetadata()

    def get_corner_coords(self):
        ulx, xres, xskew, uly, yskew, yres = self.geo_transform
        lrx = ulx + (self.src.RasterXSize * xres)
        lry = uly + (self.src.RasterYSize * yres)
        return {
            "TOP_LEFT": (ulx, uly),
            "TOP_RIGHT": (lrx, uly),
            "BOTTOM_LEFT": (ulx, lry),
            "BOTTOM_RIGHT": (lrx, lry),
        }

    def loadMetadata(self):
        # open the raster and its spatial reference
        self.src = gdal.Open(self.tif_path)

        if self.src is None:
            raise Exception('Could not load GDAL file "%s"' % self.tif_path)
        spatial_reference_raster = osr.SpatialReference(self.src.GetProjection())
        self.data = self.src.ReadAsArray().astype(float)
        # get the WGS84 spatial reference
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(4326)  # WGS84

        # coordinate transformation
        self.coordinate_transform = osr.CoordinateTransformation(
            spatial_reference, spatial_reference_raster
        )
        gt = self.geo_transform = self.src.GetGeoTransform()
        dev = gt[1] * gt[5] - gt[2] * gt[4]
        self.geo_transform_inv = (
            gt[0],
            gt[5] / dev,
            -gt[2] / dev,
            gt[3],
            -gt[4] / dev,
            gt[1] / dev,
        )

    def points_array(self):
        b = self.src.GetRasterBand(1)
        return b.ReadAsArray()

    def print_statistics(self):
        print(self.src.GetRasterBand(1).GetStatistics(True, True))

    def world_to_pixel(self, geo_matrix, x, y):
        """
        Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
        the pixel location of a geospatial coordinate
        """
        ul_x = geo_matrix[0]
        ul_y = geo_matrix[3]
        x_dist = geo_matrix[1]
        y_dist = geo_matrix[5]
        pixel = int((x - ul_x) / x_dist)
        line = -int((ul_y - y) / y_dist)
        return pixel, line

    def lookup(self, lat, lon):
        try:
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(lon, lat)
            point.Transform(self.coordinate_transform)
            px, py = self.world_to_pixel(
                self.src.GetGeoTransform(), point.GetX(), point.GetY()
            )
            v = self.data[py, px]
            return v if v != -32768 else self.SEA_LEVEL
        except Exception as e:
            print(e)
            return self.SEA_LEVEL

    def close(self):
        self.src = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class elevAPI:
    def __init__(self, path, linux=False):
        self.path = path
        self.cached_open_interfaces = []
        self.cached_open_interfaces_dict = {}
        self.linux = linux
        self.open_interfaces_size = 5
        if not os.path.exists("./" + self.path + "/summary.json"):
            self.create_summary_json()
        with open("./" + self.path + "/summary.json", "r") as f:
            self.data = json.load(f)

        self.index = index.Index()
        self.build_virtualindex()

    def build_virtualindex(self):
        index_id = 1
        for e in self.data:
            e["index_id"] = index_id
            left, bottom, right, top = (
                e["coords"][0],
                e["coords"][2],
                e["coords"][1],
                e["coords"][3],
            )
            self.index.insert(index_id, (left, bottom, right, top), obj=e)

    def loadMetadata(self):
        if self.src is None:
            raise Exception('Could not load GDAL file "%s"' % self.tif_path)
        self.spatial_reference_raster = osr.SpatialReference(self.src.GetProjection())

        # get the WGS84 spatial reference
        self.spatial_reference = osr.SpatialReference()
        self.spatial_reference.ImportFromEPSG(4326)  # WGS84

        # coordinate transformation
        self.coordinate_transform = osr.CoordinateTransformation(
            self.spatial_reference, self.spatial_reference_raster
        )
        gt = self.geo_transform = self.src.GetGeoTransform()
        dev = gt[1] * gt[5] - gt[2] * gt[4]
        self.geo_transform_inv = (
            gt[0],
            gt[5] / dev,
            -gt[2] / dev,
            gt[3],
            -gt[4] / dev,
            gt[1] / dev,
        )

    def _open_gdal_interface(self, path):
        if path in self.cached_open_interfaces_dict:
            interface = self.cached_open_interfaces_dict[path]
            self.cached_open_interfaces.remove(path)
            self.cached_open_interfaces += [path]

            return interface
        else:
            interface = GDALInterface(path)
            self.cached_open_interfaces += [path]
            self.cached_open_interfaces_dict[path] = interface

            if len(self.cached_open_interfaces) > self.open_interfaces_size:
                last_interface_path = self.cached_open_interfaces.pop(0)
                last_interface = self.cached_open_interfaces_dict[last_interface_path]
                last_interface.close()

                self.cached_open_interfaces_dict[last_interface_path] = None
                del self.cached_open_interfaces_dict[last_interface_path]

            return interface

    def create_summary_json(self):
        all_coords = []
        allfiles = [
            f
            for f in listdir(self.path)
            if isfile(join(self.path, f)) and f.endswith(".tif")
        ]
        for file in allfiles:
            full_path = join(self.path, file)
            i = self._open_gdal_interface(full_path)
            coords = i.get_corner_coords()
            all_coords += [
                {
                    "file": full_path,
                    "coords": (
                        coords["BOTTOM_RIGHT"][1],  # latitude min
                        coords["TOP_RIGHT"][1],  # latitude max
                        coords["TOP_LEFT"][0],  # longitude min
                        coords["TOP_RIGHT"][0],  # longitude max
                    ),
                }
            ]

        with open("./" + self.path + "/summary.json", "w") as f:
            json.dump(all_coords, f)

    def plotRaster(self):
        ax = plt.contourf(
            np.flipud(self.array), cmap="viridis", levels=list(range(0, 1000, 10))
        )
        cbar = plt.colorbar()
        return ax

    def getElevation(self, lat, lon):
        nearest = list(self.index.nearest((lat, lon), 1, objects=True))

        if nearest[0].object["file"] != self.path:
            self.path = nearest[0].object["file"]
            if self.linux:
                self.path = self.path.replace("\\", "/")
            self.src = gdal.Open(self.path)
            self.array = self.src.GetRasterBand(1).ReadAsArray()
            self.loadMetadata()

        xgeo, ygeo, zgeo = self.coordinate_transform.TransformPoint(lon, lat, 0)

        # convert it to pixel/line on band
        u = xgeo - self.geo_transform_inv[0]
        v = ygeo - self.geo_transform_inv[3]
        # FIXME this int() is probably bad idea, there should be half cell size thing needed
        xpix = int(self.geo_transform_inv[1] * u + self.geo_transform_inv[2] * v)
        ylin = int(self.geo_transform_inv[4] * u + self.geo_transform_inv[5] * v)
        
        v = self.array[ylin, xpix]

        return v


if __name__ == '__main__':

    username = os.getenv("USERNAME") or os.getenv("USER")
    os.environ["PROJ_LIB"] = (
        "C:/Users/" + username + "/miniconda3/envs/gdal/Library/share/proj"
    )


    elevAPI_model = "CNIG"
    elev = elevAPI(elevAPI_model)
    print(elev.getElevation(43.50328219898793, 16.442880454588995))