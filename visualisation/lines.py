import geopandas as gpd
import json
import folium
import numpy as np
from branca.colormap import LinearColormap
from shapely.geometry import LineString


# TODO: Support for multiple linestrings sent from Trace class. Multiple colors should be applied
# TODO: Extract first and last point and plot on trace with label "Start - Timestamp" - "End - Timestamp"
# TODO: Support for multiple linestrings sent from Trace class. Multiple colors should be applied
# TODO: Extract first and last point and plot on trace with label "Start - Timestamp" - "End - Timestamp"
def plot_linestring(geometry, speed_color=None, config=None):
    if (type(geometry) == LineString):
        geometry = gpd.GeoSeries(geometry)
    # set initial coords as first point of first linestring
    geometry_ = geometry.geometry[0]
    initial_coords = [geometry_.xy[1][0], geometry_.xy[0][0]]

    map_ = folium.Map(initial_coords, zoom_start=14)

    traces_json = json.loads(geometry.buffer(.00001).to_json())
    if (speed_color is not None):
        if (np.mean(speed_color) > 1):  # if absolute speeds
            red_speed, yellow_speed, blue_speed = 0, 25, 40
            cmap_caption = 'Speed Km/h'
        else:  # if relative speed
            red_speed, yellow_speed, blue_speed = 0, .4, .8
            cmap_caption = 'Relative Speed to Maximum %'

        speed_cm = LinearColormap(['red', 'yellow', 'blue'],
                                  vmin=red_speed, vmax=blue_speed + (blue_speed * .2),  # 20% maxspeed
                                  index=[red_speed, yellow_speed, blue_speed])
        speed_cm.caption = cmap_caption

        for elem, speed in zip(traces_json['features'], speed_color):
            elem['properties']['style'] = {}
            elem['properties']['style']['color'] = speed_cm(speed)
            elem['properties']['style']['fillOpacity'] = 1

        map_.add_child(speed_cm)

    traces = folium.features.GeoJson(traces_json)
    map_.add_child(traces)
    return map_
