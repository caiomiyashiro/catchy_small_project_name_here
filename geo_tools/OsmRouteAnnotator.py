import osmium
import shapely.wkb as wkblib
import numpy as np
import pandas as pd
import geopandas as gpd
from rtree import index
from shapely.geometry import Point, Polygon


# TODO: class to extract ANY desired information from OSM, not only ways. How to do it?
class OsmRouteAnnotator(osmium.SimpleHandler):

    def __init__(self, pbf_path):
        osmium.SimpleHandler.__init__(self)
        self.wkbfab = osmium.geom.WKBFactory()
        self.df = []
        self.road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'road', 'residential', 'service',
                           'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']

        print(f'loading {pbf_path}...')
        self.apply_file(pbf_path, locations=True)
        cols = ['way_id', 'nodes', 'line', 'line_length', 'name', 'maxspeed']
        self.df = pd.DataFrame(self.df, columns=cols).set_index('way_id')
        not_numeric_flag = ~self.df['maxspeed'].astype(str).str.isnumeric()
        self.df.loc[not_numeric_flag, 'maxspeed'] = '0'
        self.df['maxspeed'] = self.df['maxspeed'].astype(int)
        print('creating spatial index...')

        # Populate R-tree index with bounds of grid cells
        self.r_tree = index.Index()
        pols = []
        for way_id, row in self.df.iterrows():
            p = Polygon(row['line'].buffer(.00005).exterior.coords)
            p.maxspeed = row['maxspeed']
            p.way_id = way_id
            p.name = row['name']
            pols.append(p)

            self.r_tree.insert(way_id, p.bounds)

        self.df = gpd.GeoDataFrame(self.df, geometry=pols)
        print(f'finished')

    # TODO: Eliminate redundant ways
    # If street name has only 1 speed, keep just 1. Maybe the longest linestring?
    def process_way(self, elem):
        #  elem.nodes return a node list:
        # https://docs.osmcode.org/pyosmium/latest/ref_osm.html?highlight=noderef#osmium.osm.NodeRef

        # TagList can't be converted to dict automatically, see:
        # https://github.com/osmcode/pyosmium/issues/106
        keys = {tag.k: tag.v for tag in elem.tags}
        # filter all types of car driving highways: https://wiki.openstreetmap.org/wiki/Key:highway?uselang=en-GBs
        if (('highway' in keys.keys())):
            if (keys['highway'] in self.road_types):
                nodes = [n.ref for n in elem.nodes]
                wkb = self.wkbfab.create_linestring(elem)
                line = wkblib.loads(wkb, hex=True)
                names = [el.v for el in elem.tags if el.k == 'name']
                maxspeeds = [el.v for el in elem.tags if el.k == 'maxspeed']

                self.df.append([elem.id,
                                nodes,
                                line,
                                line.length,
                                names[0] if len(names) > 0 else '',
                                maxspeeds[0] if len(maxspeeds) > 0 else np.nan])

    def way(self, elem):
        self.process_way(elem)

    def get_street_max_speed(self, segment):
    # rank 7, segment LINESTRING (13.28866358846426 52.45759948794097, 13.28908503055573 52.45704031539945)
    # fails because of lack of precision, check out here http://arthur-e.github.io/Wicket/sandbox-gmaps3.html
    # Need mapmatch
    # Filter possible candidates using R-Tree
        idxs = list(self.r_tree.intersection(segment.bounds))
        if (len(idxs) > 0):
            # Now do actual intersection
            filter1 = self.df.loc[idxs].contains(segment)
            way_id = self.df.loc[filter1[filter1 == True].index]
            if (len(way_id) > 0):
                way_id = way_id['line_length'].idxmin()
                return self.df.loc[way_id]['maxspeed']
            else:
                first_point = Point(segment.xy[0][0], segment.xy[1][0])
                idxs = list(self.r_tree.intersection(first_point.bounds))
                if (len(idxs) > 0):
                    filter1 = self.df.loc[idxs].contains(first_point)
                    if (np.sum(filter1) > 0):
                        way_id = self.df.loc[filter1[filter1 == True].index]['line_length'].idxmin()
                        return self.df.loc[way_id]['maxspeed']

                second_point = Point(segment.xy[0][1], segment.xy[1][1])
                idxs = list(self.r_tree.intersection(second_point.bounds))
                if (len(idxs) > 0):
                    filter1 = self.df.loc[idxs].contains(second_point)
                    if (np.sum(filter1) > 0):
                        way_id = self.df.loc[filter1[filter1 == True].index]['line_length'].idxmin()
                        return self.df.loc[way_id]['maxspeed']
        raise Exception(
            f'Error mapping segment {segment} to street. Please check which segment caused it and evaluate usage of Map Matching')
