from datetime import datetime as datetime
from datetime import timedelta

import matplotlib.pyplot as plt

import statistics
import pandas as pd
import numpy as np
from .ellipses import Ellipse, EllipseList
from pandas.api.types import is_datetime64_dtype
from typing import List, Tuple
from geographiclib.geodesic import Geodesic

# matplotlib.use('TkAgg')


def __check_dist(lat1: float, lon1: float, lat2: float, lon2: float):
    d = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    return d['s12']


def proximity_interaction(df1: pd.DataFrame, df2: pd.DataFrame, minute_delay: float, distance_size: float,
                          latitude_field: str, longitude_field: str, time_field: str, id1: int, id2: int):
    intersection_list = []
    for a, row in df1.iterrows():
        lf_lat, lf_lon, lf_ts = float(row[latitude_field]), float(row[longitude_field]), row[time_field]
        min_time = lf_ts - timedelta(seconds=minute_delay * 60)
        max_time = lf_ts + timedelta(seconds=minute_delay * 60)
        sub_df2 = df2[(df2[time_field] >= min_time) & (df2[time_field] <= max_time)]
        for b, others in sub_df2.iterrows():
            if __check_dist(lf_lat, lf_lon, float(others[latitude_field]), float(others[longitude_field])) \
                    < distance_size and abs((others[time_field] - lf_ts).seconds) < minute_delay * 60:
                intersection_list.append(tuple((id1, lf_ts, lf_lat, lf_lon, id2,
                                                others[time_field], others[latitude_field],
                                                others[longitude_field])))
    return pd.DataFrame(intersection_list, columns=["Person1", "Person1_time", "Person1_lat", "Person1_lon",
                                                    "Person2", "Person2_time", "Person2_lat", "Person2_lon"])


def merge_continuous_incident_proximity(df: pd.DataFrame, id1: int, id2: int, threshold_continuous_min: float):
    p1time = df['Person1_time'].tolist()
    p2time = df['Person2_time'].tolist()
    final_start, final_end = [], []
    subq, final_sub = [], []
    i = 0
    while i < len(p1time) - 1:
        if (p1time[i + 1] - p1time[i]).total_seconds() / 60.0 <= threshold_continuous_min:
            subq.extend([p1time[i], p2time[i], p1time[i + 1], p2time[i + 1]])
        else:
            if len(subq) != 0:
                final_sub.append(subq)
            else:
                final_sub.append([p1time[i], p2time[i]])
            subq = []
        i += 1
    if len(subq) != 0:
        final_sub.append(subq)
    if len(p1time) == 1:
        final_sub.append([p1time[0], p2time[0]])
    if p1time[-1] not in final_sub[-1]:
        final_sub.append([p1time[-1], p2time[-1]])
    for item in final_sub:
        final_start.append(min(item))
        final_end.append(max(item))

    df_new = pd.DataFrame(list(zip(final_start, final_end)), columns=['Start', 'End'])
    df_new['Person1'] = id1
    df_new['Person2'] = id2
    df_new['No'] = np.arange(df_new.shape[0]) + 1
    df_new['Duration_proxi'] = df_new['End'] - df_new['Start']
    df_new['Duration_proxi'] = df_new['Duration_proxi'].dt.total_seconds().div(60)
    return df_new[['No', 'Person1', 'Person2', 'Start', 'End', 'Duration_proxi']]


def __timedifcheck(t1: pd.Timestamp, t2: pd.Timestamp):
    return abs(pd.Timedelta(t2 - t1).total_seconds())


def __check_spatial_intersect(item: Ellipse, others: Ellipse) -> bool:
    return (
            item.el[0].intersects(others.el[0])
            or item.geom.within(others.geom)
            or others.geom.within(item.geom)
    )


def __check_temporal_intersect(
        item: Ellipse, item2: Ellipse, interaction_min_delay: float
) -> bool:
    return __timedifcheck(item.t1, item2.t1) <= interaction_min_delay * 60


def get_spatiotemporal_intersect_pairs(
        ellipses_list_id1: List[Ellipse], ellipses_list_id2: List[Ellipse],
        interaction_min_delay: float, max_el_time_min: float
) -> List[Tuple[Ellipse, Ellipse]]:
    """
    Get spatially and temporally intersect PPA pairs
    :param ellipses_list_id2:
    :param ellipses_list_id1:
    :param max_el_time_min:
    :param interaction_min_delay:
    :return:
    """
    intersection_pairs = []
    for count, item in enumerate(ellipses_list_id1, 1):
        if __timedifcheck(item.t1, item.t2) > max_el_time_min * 60:
            continue  # May 15,2020: eliminate PPA if the time interval of PPA is too large:
        # if count % 500 == 0:
        # print(f"\r > On item {count} of {len(filtered_list)}", end="")

        # temporal intersect
        sub_ellipses_list = []
        for item2 in ellipses_list_id2:
            if __timedifcheck(item2.t1, item2.t2) > max_el_time_min * 60:
                continue  # eliminate PPA if the time interval of PPA of another individual is too large
            if __check_temporal_intersect(item, item2, interaction_min_delay):
                sub_ellipses_list.append(item2)

        if len(sub_ellipses_list) == 0:
            continue

        # spatial intersect
        intersection_pairs.extend(
            [
                (item, others)
                for others in sub_ellipses_list
                if __check_spatial_intersect(item, others)
            ]
        )

    return intersection_pairs


def intersect_ellipse_todataframe(intersection_df: List[Tuple[Ellipse, Ellipse]]):
    def columns_names(e: Ellipse, num: int):
        as_dict = e.to_dict()

        return {
            f"Person{num}": as_dict["pid"],
            f"Person{num}_t_start": as_dict["t2"],
            f"Person{num}_t_end": as_dict["t1"],
            f"Person{num}_startlat": as_dict["last_lat"],
            f"Person{num}_startlon": as_dict["last_lon"],
            f"Person{num}_endlat": as_dict["lat"],
            f"Person{num}_endlon": as_dict["lon"]
        }

    return pd.DataFrame(
        [
            {**columns_names(item, 1), **columns_names(item2, 2)}
            for item, item2 in intersection_df
        ]
    )


def __remove_largePPA(df: pd.DataFrame, max_el_time_min: float):
    """

    :param df:
    :param max_el_time_min:
    :return:
    """
    df['p1diff'] = df['Person1_t_end'] - df['Person1_t_start']
    df['p1diff'] = df['p1diff'].dt.total_seconds().div(60)
    df['p2diff'] = df['Person2_t_end'] - df['Person2_t_start']
    df['p2diff'] = df['p2diff'].dt.total_seconds().div(60)
    df = df[(df['p1diff'] > 0) & (df['p1diff'] < max_el_time_min)]
    df = df[(df['p2diff'] > 0) & (df['p2diff'] < max_el_time_min)]
    df = df.sort_values(by=['Person1_t_start', 'Person2_t_start'])
    return df


def __merge_continuous_incident(df: pd.DataFrame, id1: int, id2: int):
    """
    after estimating duration merge some continuous interaction incidents
    :param df:
    :param id1:
    :param id2:
    :return:
    """
    pstart = df['Start'].tolist()
    pend = df['End'].tolist()
    finalstart, finalend = [], []
    tag = []
    for i in range(0, len(pstart) - 1):
        if pend[i] >= pstart[i + 1]:
            tag.append(1)
        else:
            tag.append(0)
    tag.append(0)
    df['tag'] = tag
    finalsub, subq = [], []
    j = 0
    while j < len(tag):
        if tag[j] == 1 and tag[j + 1] == 1:
            subq.extend([pstart[j], pend[j], pstart[j + 1], pend[j + 1]])
            j += 1
        elif tag[j] == 1 and tag[j + 1] == 0:
            subq.extend([pstart[j], pend[j], pstart[j + 1], pend[j + 1]])
        else:
            if len(subq) != 0:
                finalsub.append(subq)
            else:
                finalsub.append([pstart[j], pend[j]])
            subq = []
        j += 1
    for item in finalsub:
        finalstart.append(min(item))
        finalend.append(max(item))

    df_new = pd.DataFrame(list(zip(finalstart, finalend)), columns=['Start', 'End'])
    df_new['Person1'] = id1
    df_new['Person2'] = id2
    df_new['No'] = np.arange(df_new.shape[0]) + 1
    df_new['Duration'] = df_new['End'] - df_new['Start']
    return df_new[['No', 'Person1', 'Person2', 'Start', 'End', 'Duration']]


def durationEstimator(df: pd.DataFrame, max_el_time_min: float, id1: int, id2: int):
    """
    estimate duration of interation
    :param id2:
    :param id1:
    :param df:
    :param max_el_time_min: allowable maximum time interval of PPA in minute
    :return:
    """
    df = __remove_largePPA(df, max_el_time_min)
    p1start = df['Person1_t_start'].tolist()
    p1end = df['Person1_t_end'].tolist()
    p2start = df['Person2_t_start'].tolist()
    p2end = df['Person2_t_end'].tolist()
    final_start, final_end, subsequenceOfInteraction = [], [], []
    for i in range(0, len(p1start) - 1):  # identify subsequence of continuous interaction
        if datetime.strptime(str(p1start[i]), '%Y-%m-%d %H:%M:%S') == datetime.strptime(str(p1start[i + 1]),
                                                                                        '%Y-%m-%d %H:%M:%S'):
            subsequenceOfInteraction.extend([p1start[i], p1end[i], p1end[i + 1], p2start[i], p2end[i], p2start[i + 1],
                                             p2end[i + 1]])  # append all time in a candidate pool
        elif datetime.strptime(str(p1end[i]), '%Y-%m-%d %H:%M:%S') == datetime.strptime(str(p1start[i + 1]),
                                                                                        '%Y-%m-%d %H:%M:%S'):
            subsequenceOfInteraction.extend([p1start[i], p1end[i], p1end[i + 1], p2start[i], p2end[i], p2start[i + 1],
                                             p2end[i + 1]])  # append all time in a candidate pool
        else:
            if len(subsequenceOfInteraction) == 0:
                subsequenceOfInteraction.extend([p1start[i], p1end[i], p2start[i], p2end[i]])
            final_start.append(min(subsequenceOfInteraction))  # print(i,p1start[i],p1end[i],p1start[i+1],p1end[i+1])
            final_end.append(max(subsequenceOfInteraction))
            subsequenceOfInteraction = []
    if len(subsequenceOfInteraction) != 0:
        final_start.append(min(subsequenceOfInteraction))
        final_end.append(max(subsequenceOfInteraction))
    if len(p1start) == 1:
        i = 0
        subsequenceOfInteraction.extend(
            [p1start[i], p1end[i], p2start[i], p2end[i]])  # append all time in a candidate pool
        final_start.append(min(subsequenceOfInteraction))
        final_end.append(max(subsequenceOfInteraction))
    df_new = pd.DataFrame(list(zip(final_start, final_end)), columns=['Start', 'End'])
    df_new['Person1'] = id1
    df_new['Person2'] = id2
    df_new['No'] = np.arange(df_new.shape[0]) + 1
    df_new['Duration'] = df_new['End'] - df_new['Start']
    df_new['Duration'] = df_new['Duration'].dt.total_seconds().div(60)
    return df_new[['No', 'Person1', 'Person2', 'Start', 'End', 'Duration']]





class ORTEGA:
    def __init__(
            self,
            data: pd.DataFrame,
            minute_delay: float,  # in minute
            start_time: str = None,  # only select a segment of tracking points for the two moving entities
            end_time: str = None,
            max_el_time_min: float = 10000,  # in minute
            latitude_field: str = "latitude",
            longitude_field: str = "longitude",
            id_field: str = "pid",
            time_field: str = "time_local",  # must include month, day, year, hour, minute, second

    ):
        self.data = data
        self.start_time = start_time
        self.end_time = end_time
        self.latitude_field = latitude_field
        self.longitude_field = longitude_field
        self.id_field = id_field
        self.time_field = time_field
        self.minute_delay = minute_delay
        self.max_el_time_min = max_el_time_min

        self.__validate()
        self.__start()

    @property
    def minute_delay(self):
        return self._minute_delay

    @minute_delay.setter
    def minute_delay(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("Parameter 'minute_delay' must be numeric!")
        if value <= 0:
            raise ValueError("Parameter 'minute_delay' must be greater than zero!")
        self._minute_delay = value

    @property
    def max_el_time_min(self):
        return self._max_el_time_min

    @max_el_time_min.setter
    def max_el_time_min(self, value):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("Parameter 'max_el_time_min' must be numeric!")
        if value <= 0:
            raise ValueError("Parameter 'max_el_time_min' must be greater than zero!")
        self._max_el_time_min = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Parameter 'data' must be a dataframe!")
        if value.shape[0] == 0:
            raise ValueError("The input dataframe is empty")
        self._data = value

    @property
    def latitude_field(self):
        return self._latitude_field

    @latitude_field.setter
    def latitude_field(self, value):
        if not isinstance(value, str):
            raise TypeError("Parameter 'latitude_field' must be a string!")
        if value not in self.data.columns:
            raise KeyError("Column 'latitude_field' does not exist!")
        self._latitude_field = value

    @property
    def longitude_field(self):
        return self._longitude_field

    @longitude_field.setter
    def longitude_field(self, value):
        if not isinstance(value, str):
            raise TypeError("Parameter 'longitude_field' must be a string!")
        if value not in self.data.columns:
            raise KeyError("Column 'longitude_field' does not exist!")
        self._longitude_field = value

    @property
    def id_field(self):
        return self._id_field

    @id_field.setter
    def id_field(self, value):
        if not isinstance(value, str):
            raise TypeError("Parameter 'id_field' must be a string!")
        if value not in self.data.columns:
            raise KeyError("Column 'id_field' does not exist!")
        self._id_field = value

    @property
    def time_field(self):
        return self._time_field

    @time_field.setter
    def time_field(self, value):
        if not isinstance(value, str):
            raise TypeError("Parameter 'time_field' must be a string!")
        if value not in self.data.columns:
            raise KeyError("Column 'time_field' does not exist!")
        self._time_field = value

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        if value is not None:
            try:
                datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("Incorrect 'start_time' format, should be YYYY-MM-DD HH:MM:SS")
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        if value is not None:
            try:
                datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("Incorrect 'end_time' format, should be YYYY-MM-DD HH:MM:SS")
        self._end_time = value

    def __validate(self):
        """
        private function, only can be called in side the class
        """
        if not is_datetime64_dtype(self.data[self.time_field]):
            raise TypeError("Column 'time_field' is not datetime type! Use pd.to_datetime() to convert to datetime.")

        id_list = self.data[self.id_field].unique().tolist()
        if len(id_list) != 2:
            raise ValueError(f'Only two unique id is allowed but {len(id_list)} id are found in the given dataframe!')
        else:
            self.id1, self.id2 = id_list[0], id_list[1]
            # split the dataframe according to id and filter by the time window if given
            if self.start_time is not None and self.end_time is None:
                start_time = datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')
                self.subset = self.data[[self.data[self.time_field] >= start_time]]
                self.df1 = self.subset[self.subset[self.id_field] == self.id1]
                self.df2 = self.subset[self.subset[self.id_field] == self.id2]
            elif self.start_time is None and self.end_time is not None:
                end_time = datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S')
                self.subset = self.data[[self.data[self.time_field] <= end_time]]
                self.df1 = self.subset[self.subset[self.id_field] == self.id1]
                self.df2 = self.subset[self.subset[self.id_field] == self.id2]
            elif self.start_time is not None and self.end_time is not None:
                start_time = datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')
                end_time = datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S')
                self.subset = self.data[
                    (self.data[self.time_field] >= start_time) & (self.data[self.time_field] <= end_time)]
                self.df1 = self.subset[self.subset[self.id_field] == self.id1]
                self.df2 = self.subset[self.subset[self.id_field] == self.id2]
            else:
                self.df1 = self.data[self.data[self.id_field] == self.id1]
                self.df2 = self.data[self.data[self.id_field] == self.id2]

    def __start(self):
        """
        private method, only can be called inside the class
        """

        self.ellipses_list = self.__get_ellipse_list(self.df1, self.df2)  # all ellipses for two objects
        self.ellipses_list_id1 = [i for i in self.ellipses_list if i.pid == self.id1]
        self.ellipses_list_id2 = [i for i in self.ellipses_list if i.pid == self.id2]

        #  list of intersecting ellipses
        self.all_intersection_pairs = self.__get_spatiotemporal_intersect_pairs()

        if not self.all_intersection_pairs:
            print(datetime.now(), 'Complete! No interaction found!')
        else:
            print(datetime.now(), f'Complete! {len(self.all_intersection_pairs)} pairs of interaction found!')

            # convert the list of intersecting ellipses to dataframe format
            self.df_all_intersection_pairs = intersect_ellipse_todataframe(self.all_intersection_pairs)

            # compute duration of interaction and output as a df
            self.df_duration = durationEstimator(self.df_all_intersection_pairs, self.max_el_time_min, self.id1,
                                                 self.id2)

    def __get_ellipse_list(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Construct PPA ellipse using as input the two dataframes of GPS tracks of two individuals
        :param df1: a pandas dataframe of GPS points of individual id1
        :param df2: a pandas dataframe of GPS points of individual id2
        :return:
        """
        print(datetime.now(), "Generate PPA list for the two moving entities")
        ellipses_list_gen = EllipseList(self.latitude_field, self.longitude_field, self.id_field, self.time_field)
        ellipses_list_gen.generate(df1)  # create PPA for df1
        return ellipses_list_gen.generate(df2)  # append PPA based on df2 to the above ellipses_list_gen object

    def __get_spatiotemporal_intersect_pairs(self):
        print(datetime.now(), "Getting spatial and temporal intersection pairs")
        return get_spatiotemporal_intersect_pairs(self.ellipses_list_id1, self.ellipses_list_id2,
                                                  self.minute_delay, self.max_el_time_min)

    # def __report_gen(self):
    #     """
    #     generate max value to throw out big PPA that has a length larger than 3x standard dev of all PPAs
    #     :return:
    #     """
    #     size_list = [e.el[0].length for e in self.ellipses_list]
    #
    #     return {
    #         "stdev": statistics.stdev(size_list),
    #         "mean": statistics.mean(size_list),
    #         "max_val": statistics.mean(size_list) + 3 * statistics.stdev(size_list),
    #     }

    def __compute_ppa_size(self):
        if not self.ellipses_list:
            raise ValueError("The attribute 'ellipses_list' is not found!")
        size_list1 = [e.el[0].length for e in self.ellipses_list_id1]
        size_list2 = [e.el[0].length for e in self.ellipses_list_id2]
        return {"size_list1": size_list1, "size_list2": size_list2}

    def compute_ppa_size(self, plot: bool = True):
        print(datetime.now(), "Statistics of PPA ellipses size")
        ellipse_size_collection = self.__compute_ppa_size()
        print(f"id {self.id1} ellipse length:")
        print(f"Mean:", statistics.mean(ellipse_size_collection['size_list1']))
        print(f"Min:", min(ellipse_size_collection['size_list1']))
        print(f"Max:", max(ellipse_size_collection['size_list1']))
        print(f"Median:", statistics.median(ellipse_size_collection['size_list1']))
        print(f"Standard deviation:", statistics.stdev(ellipse_size_collection['size_list1']))
        print(f"id {self.id2} ellipse length:")
        print(f"Mean:", statistics.mean(ellipse_size_collection['size_list2']))
        print(f"Min:", min(ellipse_size_collection['size_list2']))
        print(f"Max:", max(ellipse_size_collection['size_list2']))
        print(f"Median:", statistics.median(ellipse_size_collection['size_list2']))
        print(f"Standard deviation:", statistics.stdev(ellipse_size_collection['size_list2']))
        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.boxplot(ellipse_size_collection['size_list1'], labels=[self.id1], showmeans=True, showfliers=False)
            ax2.boxplot(ellipse_size_collection['size_list2'], labels=[self.id2], showmeans=True, showfliers=False)
            plt.show()
        return ellipse_size_collection

    def __compute_ppa_interval(self):
        return [
            self.df1[self.time_field].diff().dt.total_seconds().dropna(),
            self.df2[self.time_field].diff().dt.total_seconds().dropna()
        ]

    def compute_ppa_interval(self, plot: bool = True):
        print(datetime.now(), "Statistics of PPA ellipses time interval")
        time_diff = self.__compute_ppa_interval()
        print(f"id {self.id1} ellipse time interval (seconds):")
        print(f"Mean:", time_diff[0].mean())
        print(f"Min:", time_diff[0].min())
        print(f"Max:", time_diff[0].max())
        print(f"Median:", time_diff[0].median())
        print(f"Standard deviation:", time_diff[0].std())

        print(f"id {self.id2} ellipse time interval (seconds):")
        print(f"Mean:", time_diff[1].mean())
        print(f"Min:", time_diff[1].min())
        print(f"Max:", time_diff[1].max())
        print(f"Median:", time_diff[1].median())
        print(f"Standard deviation:", time_diff[1].std())

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.boxplot(time_diff[0], labels=[self.id1], showmeans=True, showfliers=False)
            ax2.boxplot(time_diff[1], labels=[self.id2], showmeans=True, showfliers=False)
            plt.show()
        return time_diff

    def plot_interaction(self, max_el_time_min: float = None, throw_out_big_ellipses: bool = True,
                         legend: bool = True, save_plot: bool = False):
        if not self.all_intersection_pairs:
            raise ValueError("The attribute 'all_intersection_pairs' is not found!")
        if not self.ellipses_list:
            raise ValueError("The attribute 'ellipses_list' is not found!")
        if max_el_time_min is None:
            max_el_time_min = self.max_el_time_min
        # max_val: float = self.__report_gen()["max_val"]
        plot_interaction(self.all_intersection_pairs, self.ellipses_list_id1, self.ellipses_list_id2,
                         self.id1, self.id2, max_el_time_min, throw_out_big_ellipses, legend, save_plot)  # max_val

    def plot_interaction_animated(self, max_el_time_min: float = None, throw_out_big_ellipses: bool = True,
                                  legend: bool = True, save_plot: bool = False):
        if not self.all_intersection_pairs:
            raise ValueError("The attribute 'all_intersection_pairs' is not found!")
        if not self.ellipses_list:
            raise ValueError("The attribute 'ellipses_list' is not found!")
        if max_el_time_min is None:
            max_el_time_min = self.max_el_time_min

        max_lon = max(self.df1[self.longitude_field].max(), self.df2[self.longitude_field].max())
        min_lon = min(self.df1[self.longitude_field].min(), self.df2[self.longitude_field].min())
        max_lat = max(self.df1[self.latitude_field].max(), self.df2[self.latitude_field].max())
        min_lat = min(self.df1[self.latitude_field].min(), self.df2[self.latitude_field].min())

        boundary = [[min_lon - 0.01, max_lon + 0.01], [min_lat - 0.01, max_lat + 0.01]]
        plot_interaction_animated(self.all_intersection_pairs, self.ellipses_list_id1, self.ellipses_list_id2, boundary,
                                  self.id1, self.id2, max_el_time_min, throw_out_big_ellipses, legend, save_plot)
        print(datetime.now(), 'Showing animated interaction...')

    def plot_original_tracks(self, max_el_time_min: float = None, throw_out_big_ellipses: bool = True,
                             legend: bool = True, save_plot: bool = False):
        if not self.ellipses_list:
            raise ValueError("The attribute 'ellipses_list' is not found!")
        if max_el_time_min is None:
            max_el_time_min = self.max_el_time_min
        # max_val: float = self.__report_gen()["max_val"]
        plot_original_tracks(self.ellipses_list_id1, self.ellipses_list_id2, self.id1, self.id2, max_el_time_min,
                             throw_out_big_ellipses, legend, save_plot)  # max_val

    def proximity(self, minute_delay: float = None, distance_size: float = 100.0):
        """

        :param minute_delay: allowable time difference between two GPS points of two individuals
        :param distance_size: define buffer size in meter
        :return:
        """
        if minute_delay is None:
            minute_delay = self.minute_delay
        return proximity_interaction(self.df1, self.df2, minute_delay, distance_size, self.latitude_field,
                                     self.longitude_field, self.time_field, self.id1, self.id2)

    def proximity_duration(self, df1: pd.DataFrame, threshold_continuous_min: float = 2):
        """

        :param df1:
        :param threshold_continuous_min: merge the subsequent interaction incidents if the time gap
        between them is less than threshold_continuous_min
        :return:
        """
        if df1.empty:
            raise ValueError("The input dataframe is empty!")
        return merge_continuous_incident_proximity(df1, self.id1, self.id2, threshold_continuous_min)
