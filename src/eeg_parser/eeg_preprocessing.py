
import argparse
from datetime import time
import io
import os
from pytictoc import TicToc
import json

from src.eeg_parser.exceptions import EEGFileNotFound, NotEEGFile
from typing import Generator, List, Tuple, Union
import pandas as pd
import numpy as np
import time
from ast import literal_eval
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from multiprocessing import Array, Process, Manager, Queue, Pool
from concurrent.futures import ProcessPoolExecutor, as_completed

NUMBER_OF_RAW_CHANNELS = 128
CSV_HEADERS = ["timestamp", "timestamp_ref", "active_modules", "xl_x","xl_y","xl_z","gyro_x","gyro_y", "gyro_z", "xl_temp","measurements"]
USED_COLS = ["timestamp", "measurements"]
BAD_IDX = [11, 36, 37, 38, 39, 47, 55, 63, 83, 87, 91, 95, 115]


# def load_measurements(filename):
#     remove_indexes = [11, 36, 37, 38, 39, 47, 55, 63, 83, 87, 91, 95, 115]
#     df = pd.read_csv(filename, skipinitialspace=True, usecols=['measurements'])
#     df['measurements'] = df['measurements'].apply(literal_eval)
#     arr = np.array(df['measurements'].values.tolist())[:, :]
#     arr = np.delete(arr, remove_indexes, 1)
#     return arr


def rectify(measurements, should_nan, threshold):
    default_val = np.NaN if should_nan else np.sign(
        measurements) * threshold
    measurements = np.where(np.abs(measurements) <
                            threshold, measurements, default_val)
    return measurements


def rereference(measurements, threshold):
    rectified = rectify(measurements, False, threshold)
    means = np.nanmean(rectified, axis=1, keepdims=True)
    return measurements - means


def butter_filt(time_series, fpower=1, cutfreq=np.array([1, 40]), sf=500, ftype='bandpass'):
    coeff = butter(fpower, cutfreq, btype=ftype, fs=sf, output='sos')
    time_series = sosfiltfilt(coeff, time_series, padlen=150, padtype="even")
    return time_series


def delete_bad_idx(arr: np.ndarray):
    return np.delete(arr, BAD_IDX, 0)

# def eeg_filter(measurements):
#     filt = butter_filt(measurements, fpower=1)
#     filt = butter_filt(filt, fpower=20, cutfreq=np.array(
#         [48, 52]), ftype='bandstop')
#     return filt


# def write_measurements(filename, measurements):
#     f = open(filename, 'wt')
#     f.write("measurements\n")
#     for measurement in measurements:
#         f.write("\"" + np.array2string(measurement,
#                 separator=",").replace('\n', '') + "\""+"\n")


class EEGFileProcessor:
    def __init__(self, eeg_file: str,
                 window_size: int,
                 window_overlap: int,
                 electrodes: list,
                 filters: List[Tuple[Union[List[int], int], int, str]] =
                 [([1, 40], 1, "bandpass"), ([48, 52], 10, "bandstop")],
                 should_rereference: bool = True, should_rectify=True, apply_hanning=True, live=True, number_of_processes=3) -> None:

        if not os.path.exists(eeg_file):
            raise EEGFileNotFound(eeg_file)
        if not eeg_file.split('.')[-1] == 'eeg':
            raise NotEEGFile(eeg_file)

        self.chosen_elec = electrodes
        self.eeg_file = eeg_file
        self.filters = filters
        self.should_rereference = should_rereference
        self.should_rectify = should_rectify
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.live = live
        self.number_of_processes = number_of_processes
        self.hanning = np.hanning(window_size) if apply_hanning else None

    def _dataframe_from_lines(self, f: io.TextIOWrapper, n_lines: int, overlap: int) -> pd.DataFrame:
        '''
        Take n_lines from the measurements column and applies the following transforms:
        1. String -> List
        2. List -> numpy array
        3. Array -> Array without disconnected sensors
        And then returns the data frame
        '''

        added = 0
        raw_lines = []
        where_overlap = None
        # Collect n_lines
        while added < n_lines:
            if added == n_lines - overlap:
                where_overlap = f.tell()
            where = f.tell()
            line = f.readline()
            wait_cycles = 0
            while line is None or line.strip() == "":
                if wait_cycles > 200:
                    return None
                time.sleep(0.001)
                line = f.readline()
                wait_cycles += 1
                continue
            # Ensure we only process full lines
            if not line.endswith("\n"):
                f.seek(where)
                continue
            raw_lines.append(line)
            added += 1

        # Parse the n lines into a dataframe using StringIO buffer
        df = pd.read_csv(io.StringIO('\n'.join(raw_lines)),
                         header=None, names=CSV_HEADERS, usecols=USED_COLS)

        if where_overlap is not None:
            f.seek(where_overlap)

        return df

    def _preprocess_measurements(self, measurements: pd.DataFrame):
        meas = np.array(measurements['measurements'].to_list())

        for filt in self.filters:
            meas = np.apply_along_axis(lambda x: butter_filt(
                x, fpower=filt[1], cutfreq=filt[0], ftype=filt[2]), 0, meas)

        threshold = 40
        if self.should_rectify:
            meas = rectify(
                meas, False, threshold)

        if self.should_rereference:
            meas = rereference(
                meas, threshold)

        if self.hanning is not None and len(self.filters)>0:
            meas = meas*self.hanning[:, None]

        measurements['measurements'] = meas.tolist()

        return measurements

    def format_preprocessing(self, q_for_df, q_for_processed_data):
        # t = TicToc()
        while True:
            df = q_for_df.get()
            # t.tic()
            serial_number = df[1]
            df = df[0]

            if df is not None:
                # Transforms
                meas = np.array([json.loads(ls) for ls in df['measurements']])
                meas = np.take(meas, self.chosen_elec, 1)
                df['measurements'] = pd.Series(meas.tolist())
                # df['measurements'] = df['measurements'].apply(
                #     literal_eval).apply(np.array).apply(delete_bad_idx)

            if df is None or df.shape[0] != self.window_size:
                return None
            # t.toc("Finished pre proc {}, time:".format(serial_number))
            q_for_processed_data.put(
                [self._preprocess_measurements(df), serial_number])

    def get_lines_of_data(self, q_for_processed_data):

        # init measurements
        base_measurements = 0
        # returns = Manager().list()
        returns = [0]*self.number_of_processes

        with open(self.eeg_file, 'rt') as f:

            not_finished = True
            format_preprocessing_pool = Pool(self.number_of_processes)
            serial_number = 0
            q_for_df = Manager().Queue()

            for _ in range(self.number_of_processes):
                format_preprocessing_pool.apply_async(
                    self.format_preprocessing, (q_for_df, q_for_processed_data))

            # Jump to the end of the file
            if self.live:
                f.seek(0, io.SEEK_END)
            f.readline()
            # t = TicToc()

            while not_finished:
                # t.tic()
                df = self._dataframe_from_lines(
                    f, self.window_size, self.window_overlap)
                # t.toc("Actual read speed")
                q_for_df.put([df, serial_number])
                if df is None:
                    format_preprocessing_pool.close()
                    format_preprocessing_pool.join()
                    not_finished = False
                    break
                serial_number += 1

    def start_parsing(self) -> Generator[pd.DataFrame, None, None]:

        q_for_processed_data = Manager().Queue()

        # init the reading process
        reading_process = Process(
            target=self.get_lines_of_data, args=(q_for_processed_data,))
        reading_process.start()

        # Continuos fill with overlap
        serial_number = 0  # current processed data
        dict_of_processed_data = {}
        wait_counter = time.time()
        while True:
            if q_for_processed_data.empty():
                if time.time() - wait_counter >= 60:
                    break
                continue
            wait_counter = time.time()

            (processed_data, id) = q_for_processed_data.get()
            dict_of_processed_data[id] = processed_data
            if serial_number in dict_of_processed_data:
                # print(dict_of_processed_data[serial_number])
                yield dict_of_processed_data[serial_number]
                dict_of_processed_data.pop(serial_number)
                serial_number += 1

    def proc(self):
        parsing_proc = Process(target=self.start_parsing)
        parsing_proc.start()
        gener = parser.start_parsing()
        for window in gener:
            print(window)

if __name__ == "__main__":

    parser = EEGFileProcessor("C:/Users/ofera/Studies/space_project/sensi/data/2021-10-07-14-08-25/2021-10-07-14-10-52.eeg",
                              window_overlap=100, window_size=300, live=True, electrodes=[60, 67, 95])


    parser.proc()
    # parsing_proc = Process(target=parser.start_parsing)
    # parsing_proc.start()

    # # t = TicToc()
    # gener = parser.start_parsing()
    # # data = parser.get_parsed_data()
    #
    # for window in gener:
    #     print(window)

    # t.tic()
    # val = next(gener)
    # t.toc()
    #
    # t.tic()
    # val = next(gener)
    # t.toc()

    print("Finished testing EEGProcessor")

