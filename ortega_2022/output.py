from osgeo import ogr
from .ellipses import Ellipse
from typing import List
import pandas as pd
import statistics
import matplotlib.pyplot as plt


def output_shapefile(ellipses_list: List[Ellipse], max_el_time_min: float, id1, id2):
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(f"{id1}_{id2}.shp")
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('pid', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('lat', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('lon', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('time', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('last_lat', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('last_lon', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('last_time', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('ppa_id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    i = 0
    for item in ellipses_list:
        if abs(pd.Timedelta(item.t2 - item.t1).total_seconds()) > max_el_time_min * 60:
            continue
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        i += 1
        feat.SetField('pid', item.pid)
        feat.SetField('lat', item.lat)
        feat.SetField('lon', item.lon)
        feat.SetField('last_lat', item.last_lat)
        feat.SetField('last_lon', item.last_lon)
        feat.SetField('time', str(item.t1))
        feat.SetField('last_time', str(item.t2))
        # feat.SetField('time', item.t1.year, item.t1.month, item.t1.day, item.t1.hour, item.t1.minute, item.t1.second, 0)
        # feat.SetField('last_time', item.t2.year, item.t2.month, item.t2.day, item.t2.hour, item.t2.minute, item.t2.second, 0)
        feat.SetField('ppa_id', i)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(item.geom.wkb)
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = geom = None  # destroy these
    # Save and close everything
    ds = layer = feat = geom = None


def compute_ppa_size(ellipses_list1: List[Ellipse], ellipses_list2: List[Ellipse], id1, id2, plot: bool = True):
    print("Statistics of PPA ellipses size")
    size_list1 = [e.el[0].length for e in ellipses_list1]
    size_list2 = [e.el[0].length for e in ellipses_list2]

    ellipse_size_collection = {"size_list1": size_list1, "size_list2": size_list2}
    print(f"id {id1} ellipse length:")
    print(f"Mean:", statistics.mean(ellipse_size_collection['size_list1']))
    print(f"Min:", min(ellipse_size_collection['size_list1']))
    print(f"Max:", max(ellipse_size_collection['size_list1']))
    print(f"Median:", statistics.median(ellipse_size_collection['size_list1']))
    print(f"Standard deviation:", statistics.stdev(ellipse_size_collection['size_list1']))
    print(f"id {id2} ellipse length:")
    print(f"Mean:", statistics.mean(ellipse_size_collection['size_list2']))
    print(f"Min:", min(ellipse_size_collection['size_list2']))
    print(f"Max:", max(ellipse_size_collection['size_list2']))
    print(f"Median:", statistics.median(ellipse_size_collection['size_list2']))
    print(f"Standard deviation:", statistics.stdev(ellipse_size_collection['size_list2']))
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.boxplot(ellipse_size_collection['size_list1'], labels=[id1], showmeans=True, showfliers=False)
        ax2.boxplot(ellipse_size_collection['size_list2'], labels=[id2], showmeans=True, showfliers=False)
        plt.show()
    return ellipse_size_collection


def compute_ppa_interval(df1: pd.DataFrame, df2: pd.DataFrame, time_field: str, id1, id2, plot: bool = True):
    print("Statistics of PPA ellipses time interval")
    time_diff = [
        df1[time_field].diff().dt.total_seconds().dropna(),
        df2[time_field].diff().dt.total_seconds().dropna()
    ]
    print(f"id {id1} ellipse time interval (seconds):")
    print(f"Mean:", time_diff[0].mean())
    print(f"Min:", time_diff[0].min())
    print(f"Max:", time_diff[0].max())
    print(f"Median:", time_diff[0].median())
    print(f"Standard deviation:", time_diff[0].std())

    print(f"id {id2} ellipse time interval (seconds):")
    print(f"Mean:", time_diff[1].mean())
    print(f"Min:", time_diff[1].min())
    print(f"Max:", time_diff[1].max())
    print(f"Median:", time_diff[1].median())
    print(f"Standard deviation:", time_diff[1].std())

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.boxplot(time_diff[0], labels=[id1], showmeans=True, showfliers=False)
        ax2.boxplot(time_diff[1], labels=[id2], showmeans=True, showfliers=False)
        plt.show()
    return time_diff
