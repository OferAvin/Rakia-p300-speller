from os import path
from queue import Empty
import time
from typing import List
from numpy import floor
import numpy
import json
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np
from multiprocessing import Queue
from consts import EEG_HEADERS, EEG_USED_COLS, EVALUATION_LOG_NAME, SAMPLING_RATE, TRAINING_LOG_NAME, WINDOW_SIZE


def extract_first_timestamp(eeg_file) -> float:
    f = open(eeg_file, 'rt')
    f.readline()  # skip headers
    line = f.readline()
    f.close()
    return float(line.split(',')[0])


class SpellerModel():
    def __init__(self, eeg_file: str, speller_queue: Queue, save_raw_data: bool) -> None:
        self.eeg_file = eeg_file
        self.speller_queue = speller_queue
        self.first_timestamp = extract_first_timestamp(eeg_file)
        self.model = LDA()
        self.save_raw_data = save_raw_data

        self.training_log_file = path.join(
            path.dirname(eeg_file), TRAINING_LOG_NAME)
        self.evaluation_log_file = path.join(
            path.dirname(eeg_file), EVALUATION_LOG_NAME)

    def get_data_for_flash(self, timestamp) -> pd.DataFrame:
        # \Delta Timestamp * Sample rate [hz] / 1000 (to scale for ms) - 2 (to account for header and redundant line )
        num_skip = int((timestamp - self.first_timestamp)
                       * SAMPLING_RATE/1000) - 2
        print(num_skip)
        # \Desired sample time (in ms) * Sample rate [hz] / 1000 (to scale for ms)
        num_rows = WINDOW_SIZE * SAMPLING_RATE // 1000
        df = pd.read_csv(self.eeg_file, skiprows=num_skip, nrows=num_rows, names=EEG_HEADERS,
                         header=0, usecols=EEG_USED_COLS, converters={
                             "measurements": lambda x: json.loads(x)
                         })
        if df.empty:
            raise Exception("Couldn't match flash timestamp to eeg")
        return np.array(df.measurements.to_list())

    def tail_file(self, file):
        f = open(file, 'rt')
        f.readline()  # skip header
        end_reached = False
        counter = 0
        while True:
            where = f.tell()
            line = f.readline()
            while line is None or line.strip() == "":
                time.sleep(0.1)
                line = f.readline()
                counter += 1
                if end_reached and counter >= 10:
                    return None
            # Ensure we only process full lines
            if not line.endswith("\n"):
                f.seek(where)
                continue
            counter = 0
            end_reached = yield line

    def run(self):
        # Await permission to train
        training_line_generator = self.tail_file(self.training_log_file)
        should_finish = False
        data = []
        # training_data = pd.read_csv(self.training_log_file, header=0, names=['X','y'], converters={

        print("Grabbing training data")
        while True:
            try:
                self.speller_queue.get_nowait()
                should_finish = True
            except Empty:
                pass
            try:
                line = training_line_generator.send(should_finish)
            except StopIteration:
                break
            ts, label = line.split(',')
            data.append((self.get_data_for_flash(float(ts)*1000), label))

        training_data = pd.DataFrame(data, columns=['X', 'y'])
        # 'X': lambda x: self.get_data_for_flash(float(x)*1000)})
        if self.save_raw_data:
            training_data.to_pickle(self.eeg_file + "training_data.pkl")
            print("Saved training data")

        # TRAIN THE MODEL HERE

        self.speller_queue.put("Model trained")

        # PREDICTION LOOP
        eval_line_generator = self.tail_file(self.evaluation_log_file)
        is_ready = False
        while True:
            try:
                self.speller_queue.get_nowait()
                is_ready = True
            except Empty:
                pass
            try:
                line = training_line_generator.send(False)
            except StopIteration:
                break
            ts, label = line.split(',')
            data.append((self.get_data_for_flash(float(ts)*1000), label))

            if is_ready:
                # perform prediction
                self.speller_queue.put("batata")
                is_ready = False
