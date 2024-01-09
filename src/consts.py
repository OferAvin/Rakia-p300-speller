LETTERS = {
    (1, 6): "a", (1, 7): "b", (1, 8): "c", (1, 9): "d", (1, 10): "e", (1, 11): "f",
            (2, 6): "g", (2, 7): "h", (2, 8): "i", (2, 9): "j", (2, 10): "k", (2, 11): "l",
            (3, 6): "m", (3, 7): "n", (3, 8): "o", (3, 9): "p", (3, 10): "q", (3, 11): "r",
            (4, 6): "s", (4, 7): "t", (4, 8): "u", (4, 9): "v", (4, 10): "w", (4, 11): "x",
            (5, 6): "y", (5, 7): "z", (5, 8): "space", (5, 9): "delete", (5, 10): "send", (5, 11): "iss"

}

TRAINING_LOG_NAME = "speller_training.csv"
EVALUATION_LOG_NAME = "speller_evaluation.csv"
SAMPLING_RATE = 500 # Hz

WINDOW_SIZE = 450 # ms
EEG_HEADERS = ["timestamp","timestamp_ref","active_modules","xl_x","xl_y","xl_z","gyro_x","gyro_y","gyro_z","xl_temp","measurements"]
EEG_USED_COLS = ["timestamp", "measurements"]

SENSI_DIR = "/home/yoni/Devel/sensi/data/"