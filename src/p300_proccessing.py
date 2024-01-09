import sys
import os
from consts import SENSI_DIR
from model import SpellerModel
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
import src.eeg_parser.eeg_preprocessing as eeg_parser
from src.p300speller.p300_speller import MainWindow

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier as gb
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from numpy import mean, std


import time
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mne import filter
from enum import IntEnum
from math import floor, ceil
import pickle
from multiprocessing import Process, Manager, Queue
from tkinter import Tk, Toplevel, Text, END, Canvas, PhotoImage
from PIL import ImageTk, Image


def queue_filler(q):
    time.sleep(25)
    print("sending shit")
    q.put("just a test")
    time.sleep(5)
    q.put("x")
    time.sleep(5)
    q.put("xi")
    time.sleep(5)
    q.put("xij")

class Elec(IntEnum):
    Afz = 60
    Fz = 67
    Fcz = 95


class P300Speller(object):
    def __init__(self,
                 flash_timing_log_file: str,
                 flash_duration: int,
                 break_duration: int,
                 interval: int):
        self.flash_timing_log_file = os.path.join(os.path.dirname(__file__), flash_timing_log_file)
        
        self.flash_duration = flash_duration
        self.break_duration = break_duration
        self.interval = interval
        self.root = None
        self.config_window = None
        self.p300_window = None
        self.instructions_window = None
        self.images_path = None
        self.canvas = None
        self.img = None

        self.instruction = ""
        self.rows = [1, 2, 3, 4, 5]
        self.cols = [6, 7, 8, 9, 10, 11]

        self.letters = {
            (1, 6): "a", (1, 7): "b", (1, 8): "c", (1, 9): "d", (1, 10): "e", (1, 11): "f",
            (2, 6): "g", (2, 7): "h", (2, 8): "i", (2, 9): "j", (2, 10): "k", (2, 11): "l",
            (3, 6): "m", (3, 7): "n", (3, 8): "o", (3, 9): "p", (3, 10): "q", (3, 11): "r",
            (4, 6): "s", (4, 7): "t", (4, 8): "u", (4, 9): "v", (4, 10): "w", (4, 11): "x",
            (5, 6): "y", (5, 7): "z", (5, 8): "space", (5, 9): "delete", (5, 10): "send", (5, 11): "iss",

        }

    def speller_init(self, prediction_queue):
        self.root = Tk()
        self.config_window = MainWindow(self.root, self.flash_timing_log_file)
        self.images_path = self.config_window.config.flash_image_path
        self.p300_window = self.config_window.open_p300_window(prediction_queue)
        self.config_window.config.flash_duration.set(self.flash_duration)
        self.config_window.config.break_duration.set(self.break_duration)

    def start_training(self, targets):
        self.p300_window.start_training(targets)

    def start_evaluation(self):
        self.p300_window.start_evaluation()

    def run_speller(self):
        # self.p300_window.master.after(0, self.p300_window.start)
        # self.p300_window.master.after(self.interval, self.p300_window.pause)
        # self.p300_window.master.after(self.interval, self.root.quit)
        self.root.mainloop()

    def close_speller(self):
        self.root.destroy()

    def remove_log_file(self):
        os.remove(self.flash_timing_log_file)


class P300Processor(object):
    def __init__(self, is_online: bool,
                 sampling_rate: int,
                 window_size_ms: int,
                 speller: P300Speller,
                 eeg_file=None):
        self.prediction_queue = Queue()
        self.sensi_dir = SENSI_DIR

        self.speller = speller

        self.is_online = is_online
        self.eeg_file = eeg_file if eeg_file else self.get_eeg_filepath()
        self.eeg_data = Queue()
        self.last_eeg_window = pd.DataFrame()
        self.relevant_training_data = []
        self.tagging_for_training_data = []

        self.rows_vote_count = []
        self.cols_vote_count = []

        # eeg settings
        self.sampling_rate = sampling_rate
        self.relevant_window_size_ms = window_size_ms
        self.parser_window_size_ms = window_size_ms * 2
        self.relevant_window_size_tp = ceil(self.relevant_window_size_ms * self.sampling_rate / 1000)  # tp = time point
        self.parser_window_size_tp = ceil(self.parser_window_size_ms * self.sampling_rate / 1000)
        self.max_trigger_range_ms = 10
        self.max_trigger_range_tp = floor(self.max_trigger_range_ms * self.sampling_rate / 1000)

        self.l_freq = 2
        self.h_freq = 30
        self.electrodes = [Elec.Afz, Elec.Fcz, Elec.Fz]
        print("window_size_tp:", self.parser_window_size_tp, "window_ol_tp:", self.relevant_window_size_tp)
        # self.parser = eeg_parser.EEGFileProcessor(eeg_file=self.eeg_file, window_overlap=self.relevant_window_size_tp,
        #                                           window_size=self.parser_window_size_tp, live=self.is_online,
        #                                           electrodes=self.electrodes)

        self.predictor = SpellerModel(self.eeg_file, self.prediction_queue, True)

        # model settings
        # self.model = lda()
        self.model = gb()
        # self.model = svm.SVC()
        self.n_splits = 10
        self.n_repeats = 5

        self.main()

    def get_eeg_filepath(self):
        all_subdirs = [os.path.join(self.sensi_dir, d) for d in os.listdir(self.sensi_dir)
                       if os.path.isdir(os.path.join(self.sensi_dir, d))]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        for file in os.listdir(latest_subdir):
            if file.endswith(".eeg"):
                return os.path.join(latest_subdir, file)

    def run_parsing(self, started_parsing):
        gener = self.parser.start_parsing()
        for window in gener:
            # print(window) #not asking the 0\1 question because of the thread
            started_parsing.value = True
            self.eeg_data.put(window)

    def parse_flash_logs(self, df):
        letter = int(df['letter'][df['letter'].find("[") + 1:df['letter'].find("]")])
        flash_time = df['timing']
        return letter, flash_time

    def save_model(self):
        with open('p300Model.pkl', 'wb') as outp:
            pickle.dump(self.model, outp, pickle.HIGHEST_PROTOCOL)

    def train_model(self):
        print("start training")
        cv = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=1)
        scores = cross_val_score(self.model, scoring='accuracy', cv=cv, n_jobs=2,
                                 X=self.relevant_training_data, y=self.tagging_for_training_data, verbose=1)

        print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    # def train_model(self):
    #     print("start training")
    #
    #     print(self.tagging_for_training_data)
    #     print(self.tagging_for_training_data.count(True)/len(self.tagging_for_training_data))
    #
    #     self.model.fit(X=self.relevant_training_data, y=self.tagging_for_training_data)
    #     clf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=1)
    #     scores = cross_val_score(self.model, scoring='accuracy', cv=cv, n_jobs=2,
    #                              X=self.relevant_training_data, y=self.tagging_for_training_data, verbose=2)
    #
    #     print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

    def get_next_relevant_eeg_window(self, flash_time: float):  # might be good for online too
        eeg_window = self.last_eeg_window if not self.last_eeg_window.empty else self.eeg_data.get()
        while True:
            relevant = flash_time > eeg_window["timestamp"].iloc[0] \
                       and flash_time + self.relevant_window_size_ms - 5 < eeg_window["timestamp"].iloc[-1]
            gohst_flash = flash_time < eeg_window["timestamp"].iloc[0]
            # print("start", eeg_window["timestamp"].iloc[0])
            # print("flash", flash_time)
            # print("end", eeg_window["timestamp"].iloc[-1])
            # print("time from start:", flash_time - eeg_window["timestamp"].iloc[0])
            # print("time to end:", eeg_window["timestamp"].iloc[-1] - flash_time - 450)
            if relevant:
                break
            elif gohst_flash:
                return None
            eeg_window = self.eeg_data.get()

        # print("time:", flash_time, "\n", eeg_window["timestamp"].iloc[0], eeg_window["timestamp"].iloc[-1])
        for idx, timestamp in enumerate(eeg_window["timestamp"].tolist()):
            if timestamp < flash_time < timestamp + self.max_trigger_range_ms:
                self.last_eeg_window = eeg_window
                return eeg_window.iloc[idx:idx + self.relevant_window_size_tp]
            continue

    def get_vector_for_each_elec(self, eeg_window: pd.DataFrame):
        # print("relevant windows: ", eeg_window)
        arr = eeg_window.explode('measurements')['measurements'].to_numpy()
        # print("1dim arr: ", arr, "len: ", len(arr))
        a = np.empty((0, self.relevant_window_size_tp))
        print(arr)
        print(arr.shape)
        print(eeg_window.shape)
        for j in range(len(self.electrodes)):
            a = np.vstack((a, arr[j::len(self.electrodes)]))

        return a

    def exec_preprocess(self, relevant_window):
        elec_matrix = self.get_vector_for_each_elec(relevant_window)
        elec_matrix = elec_matrix.astype(float)
        ## Bandpass for elec_matrix
        # print("elec mat:", elec_matrix)
        # print("elec mat shape:", elec_matrix.shape)
        # print("elec mat type:", type(elec_matrix))
        processed_window = filter.filter_data(data=elec_matrix, verbose=0, l_freq=self.l_freq,
                                              h_freq=self.h_freq, sfreq=self.sampling_rate)
        return processed_window.flatten()

    def training_preprocess_and_tagging(self, target):
        flashes_df = pd.read_csv(self.speller.flash_timing_log_file, names=('letter', 'timing',), delimiter=" ", )
        for index, row in flashes_df.iterrows():
            letter, flash_time = self.parse_flash_logs(row)
            relevant_window = self.get_next_relevant_eeg_window(flash_time * 1000)
            if relevant_window:
                processed_window = self.exec_preprocess(relevant_window)
                self.relevant_training_data.append(processed_window)
                tag = True if letter == target[0] or letter == target[1] else False
                self.tagging_for_training_data.append(tag)

        print(len(self.relevant_training_data), len(self.tagging_for_training_data))

    def exec_online_spelling(self):
        flashes_df = pd.read_csv(self.speller.flash_timing_log_file, names=('letter', 'timing',), delimiter=" ", )
        for index, row in flashes_df.iterrows():
            letter, flash_time = self.parse_flash_logs(row)
            relevant_window = self.get_next_relevant_eeg_window(flash_time * 1000)
            if relevant_window:
                processed_window = self.exec_preprocess(relevant_window)
                prediction = self.model.predict(processed_window)
                if prediction:
                    if letter in self.speller.rows:
                        self.rows_vote_count.append(letter)
                    else:
                        self.cols_vote_count.append(letter)
        return (max(self.rows_vote_count, key=self.rows_vote_count.count),
                max(self.cols_vote_count, key=self.cols_vote_count.count))

    def average_training_data(self, n_trials):
        true_data = [val for idx, val in enumerate(self.relevant_training_data)
                     if self.tagging_for_training_data[idx]]
        false_data = [val for idx, val in enumerate(self.relevant_training_data)
                      if not self.tagging_for_training_data[idx]]

    def average_every_n_elements_in_list(self, ls: list, n: int):
        tail_avg = None
        if len(ls) % n != 0:
            new_len = (len(ls)//n)*n
            list_tail = ls[new_len:]
            tail_avg = np.mean(list_tail)
            ls = ls[0:new_len]
        avg_ls = np.mean(np.array(ls).reshape(-1, n), axis=1)
        if tail_avg:
            avg_ls = np.append(avg_ls, tail_avg)
        return avg_ls

    def plot_ERP(self):
        falseERP_idx = [i for i, x in enumerate(self.tagging_for_training_data) if x == False]
        trueERP_idx = [i for i, x in enumerate(self.tagging_for_training_data) if x == True]

        falseData = [self.relevant_training_data[i] for i in falseERP_idx]
        trueData = [self.relevant_training_data[i] for i in trueERP_idx]

        falseERP = sum(falseData) / len(falseERP_idx)
        trueERP = sum(trueData) / len(trueERP_idx)

        plt.figure("First electrode")
        plt.plot(list(range(225)), falseERP[0:225],
                 list(range(225)), trueERP[0:225])
        plt.figure("Second electrode")
        plt.plot(list(range(225)), falseERP[225:450],
                 list(range(225)), trueERP[225:450])
        plt.figure("Third electrode")
        plt.plot(list(range(225)), falseERP[450:],
                 list(range(225)), trueERP[450:])
        plt.show()

    def main(self):
        # started_parsing = Manager().Value("b", False)
        # parsing_proc = Process(target=self.run_parsing, args=(started_parsing,))
        # parsing_proc.start()
        # while not started_parsing.value:
        #     time.sleep(0.1)


        letters = self.speller.letters
        targets = []
        for i in range(3):
            target = (random.choice(self.speller.rows), random.choice(self.speller.cols))
            targets.append(target)
        p = Process(target=self.predictor.run,)
        p.start()

        self.speller.speller_init(self.prediction_queue)
        self.speller.start_training(targets)
        self.speller.run_speller()
        # for target in targets:
        #     self.speller.set_instruction("Please focus on the letter {} and press start".format(letters[target]))
        #     self.speller.run_speller()
        #     time.sleep(10)
        #     # try:
        #     #     result = exec_operation(target)
        #     # except NotFittedError as e:
        #     #     print(repr(e), "\nTraining or Loading a model should be done first")
        #     # self.speller.remove_log_file()

        # self.train_model()
        # time.sleep(5)
        # self.speller.start_evaluation()
        # input("READYYYY")
        self.speller.close_speller()
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nUI is closed\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")


if __name__ == '__main__':
    eeg_file = "/home/yoni/Devel/sensi/data/2021-10-19-19-46-36/2021-10-19-19-47-31.eeg"
    timing_log = os.path.dirname(eeg_file)
    p300_speller = P300Speller(flash_timing_log_file=timing_log,
                               flash_duration=65, break_duration=130, interval=15000)
    main_process = P300Processor(is_online=False,
                                 eeg_file=None,
                                 window_size_ms=450, sampling_rate=500,
                                 speller=p300_speller)

    # main_process = P300Processor(is_online=True, sampling_rate=500,
    #                              window_size_ms=450, speller=p300_speller)
