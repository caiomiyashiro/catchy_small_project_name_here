import pandas as pd
import numpy as np
import geopandas as gpd
from geo_tools.geo_distance import np_distance_haversine
from shapely.geometry import LineString
from visualisation.lines import plot_linestring


# --------------------------------------------------#
# TODO: maybe think on standard formats like GPX?  #
# --------------------------------------------------#
class Trace():

    def __init__(self, trace=None, osm_route_annotator=None):
        # process pbf file here for maxspeeds?
        self.raw_trace = None
        self.df = None
        self.traces_seg_idx = None
        self.segment_pairs = None
        self.route_annotator = osm_route_annotator

        self.__read_process_gps(trace)

    def __create_segment_agg_data(self):
        # linestring geometries
        group_lambda = lambda group: LineString(
            [[lon, lat] for lon, lat in zip(group['coordinatelongitude'], group['coordinatelatitude'])])
        self.traces_seg_idx = gpd.GeoDataFrame(self.traces_seg_idx,
                                               geometry=self.df.groupby('trace_rank').apply(group_lambda).values)

        # avg speed array
        avg_speed_lambda = lambda group: group['avg_speed_km_h'].values
        self.traces_seg_idx['avg_speed_km_h'] = self.df.groupby('trace_rank').apply(avg_speed_lambda).values

        # avg acceleration
        avg_acceleration_lambda = lambda group: group['avg_acceleration_m_s'].values
        self.traces_seg_idx['avg_acceleration_m_s'] = self.df.groupby('trace_rank').apply(
            avg_acceleration_lambda).values

        # create linestrings of 2 points for segment analysis
        points_lambda = lambda group: [LineString([[lon1, lat1], [lon2, lat2]]) for lon1, lat1, lon2, lat2
                                       in zip(group['coordinatelongitude'][:-1], group['coordinatelatitude'][:-1],
                                              group['coordinatelongitude'][1:], group['coordinatelatitude'][1:])]
        self.traces_seg_idx['segments'] = self.df.groupby('trace_rank').apply(points_lambda).values

        timestamp_lambda = lambda group: group['lastupdate'].values
        self.traces_seg_idx['timestamps'] = self.df.groupby('trace_rank').apply(timestamp_lambda).values

        accuracy_lambda = lambda group: group['accuracy'].values
        self.traces_seg_idx['radiuses'] = self.df.groupby('trace_rank').apply(accuracy_lambda).values

    def __create_segment_idx(self, min_size_trace=5, max_size_trace=100, max_time_window_seg=30):
        """
        # segment traces based on:
        # 1 - add rules here
        """

        # find and split trace when too apart - time between points > max_time_window_seg
        self.df['trace_rank'] = np.zeros(self.df.shape[0])
        inv_timediff = self.df['diff_time_sec'] > max_time_window_seg
        self.df.loc[inv_timediff, 'trace_rank'] = 1

        # segment trace by shift change
        changed_status = self.df['onlinestatus'] != self.df['onlinestatus'].shift(1)
        self.df.loc[changed_status, 'trace_rank'] = 1
        self.df.loc[0, 'trace_rank'] = 0

        # find and split big traces > max_size_trace
        trace_ix = self.df.loc[self.df['trace_rank'] == 1].index
        next_trace_ix = np.array(trace_ix.tolist()[1:] + [trace_ix.values[-1]])
        diff_len_ix = next_trace_ix - trace_ix
        is_big_trace_ix = np.where(diff_len_ix > max_size_trace)[0]
        for ix_ in is_big_trace_ix:
            self.df.loc[np.arange(trace_ix[ix_], trace_ix[ix_ + 1], max_size_trace), 'trace_rank'] = 1

        # create segments
        self.df['trace_rank'] = self.df['trace_rank'].cumsum()

        # check segment size - if segment size > min_size_trace
        valid_traces = self.df['trace_rank'].value_counts()
        valid_traces = valid_traces[valid_traces > min_size_trace].index.values
        self.df = self.df.loc[self.df['trace_rank'].isin(valid_traces)].copy().reset_index(drop=True)
        self.df['trace_rank'] = self.df['trace_rank'].rank(method='dense')

        # rename and filter fields for formatting
        traces_overview = self.df['trace_rank'].drop_duplicates(keep='first').index
        self.traces_seg_idx = self.df.loc[traces_overview][['onlinestatus', 'trace_rank']].reset_index(drop=True)

        # create linestrings for each trace segment
        self.__create_segment_agg_data()

        print(f'{self.traces_seg_idx.shape[0]} trace segments created')

    def __filter_bad_points(self, speed_lim_km_h=100, accel_lim_km_h=10):
        # remove all POOLING:false status
        self.df = self.df.loc[self.df['locationforfleettypeupdated'] != 'POOLING:false']

        # removed duplicated timestamps
        no_duplicates_ix = self.df['lastupdate'].drop_duplicates().index
        self.df = self.df.loc[no_duplicates_ix].reset_index(drop=True)

        lat_lon2 = self.df[['coordinatelatitude', 'coordinatelongitude', 'lastupdate']].copy()

        # calculate and filter based on unreal speed and acceleration
        lat_lon2 = lat_lon2.merge(lat_lon2.shift(-1), how='left', left_index=True, right_index=True).iloc[:-1]

        lat_lon2['distance_meters'] = np_distance_haversine(lat_lon2['coordinatelatitude_x'],
                                                            lat_lon2['coordinatelongitude_x'],
                                                            lat_lon2['coordinatelatitude_y'],
                                                            lat_lon2['coordinatelongitude_y']) * 1000

        lat_lon2['diff_time_sec'] = (lat_lon2['lastupdate_y'] - lat_lon2['lastupdate_x']).dt.seconds

        lat_lon2['avg_speed_km_h'] = (lat_lon2['distance_meters'] / lat_lon2['diff_time_sec']) * 3.6

        lat_lon2['diff_avg_speed_km_h'] = lat_lon2['avg_speed_km_h'].diff(1)

        lat_lon2['avg_acceleration_m_s'] = (lat_lon2['diff_avg_speed_km_h'] / lat_lon2['diff_time_sec']) / 3.6

        copy_cols_name = ['distance_meters', 'diff_time_sec', 'avg_speed_km_h', 'diff_avg_speed_km_h',
                          'avg_acceleration_m_s']
        self.df[copy_cols_name] = lat_lon2[copy_cols_name]

        ### filter
        ix1 = lat_lon2.loc[np.abs(lat_lon2['avg_speed_km_h']) < speed_lim_km_h].index.values
        ix2 = lat_lon2.loc[lat_lon2['avg_acceleration_m_s'] < accel_lim_km_h].index.values
        ix_valid = list(set(ix1.tolist() + ix2.tolist()))

        self.df = self.df.loc[ix_valid].copy().reset_index(drop=True)

    def __read_process_gps(self, path):
        self.raw_trace = pd.read_csv(path).sort_values('lastupdate').reset_index(drop=True)
        self.raw_trace['lastupdate'] = pd.to_datetime(self.raw_trace['lastupdate'])
        self.raw_trace[['locationforfleettypeupdated', 'onlinestatus']] = self.raw_trace[
            ['locationforfleettypeupdated', 'onlinestatus']].astype('category')
        self.raw_trace.drop(['currentfleettypes', 'activefleettypes',
                             'outdatedfleettypes', 's2cell'], axis=1, inplace=True)
        self.df = self.raw_trace.copy()

        # remove phisically wrong datapoints
        self.__filter_bad_points()
        # segment traces based on rules
        self.__create_segment_idx()

    def get_trace_info(self, rank=None):
        if (rank is None):
            return self.traces_seg_idx
        else:
            filtered = self.traces_seg_idx.loc[self.traces_seg_idx['trace_rank'] == rank].loc[rank - 1]
            lon, lat = filtered['geometry'].xy
            timestamps = filtered['timestamps']
            radiuses = filtered['radiuses']

            return lat, lon, timestamps, radiuses

    def plot(self, trace_rank=None, speed_color=None):
        if (trace_rank is None):
            # TODO: If no rank is sent, we should plot all traces
            geoms = self.traces_seg_idx
        else:
            geoms = self.traces_seg_idx.loc[self.traces_seg_idx['trace_rank'] == trace_rank]

        fake_values = np.arange(len(geoms['segments'].values[0]))
        trace = gpd.GeoDataFrame(fake_values, geometry=geoms['segments'].values[0])
        if (speed_color == 'absolute'):
            speeds = geoms['avg_speed_km_h'].values[0]
        elif (speed_color == 'relative'):
            if (self.route_annotator is not None):
                car_speeds = geoms['avg_speed_km_h'].values[0]
                speeds = []
                for seg in geoms['segments'].values[0]:
                    speeds.append(self.route_annotator.get_street_max_speed(seg))
                speeds = np.where(speeds == 0, 1, speeds)
                speeds = car_speeds[1:] / speeds

            else:
                raise Exception('Attribute route annotator not set. Please check constructor variables')
        else:
            fake_values = [1]
            trace = gpd.GeoDataFrame(fake_values, geometry=geoms['geometry'].values)
            speeds = None
        self.t = speeds
        return plot_linestring(trace, speeds)

    @staticmethod
    def latlon2linestring(lat, lon):
        return LineString([[lon, lat] for lat, lon in zip(lat, lon)])

    @staticmethod
    def linestring2latlon(linestring):
        return linestring.xy
