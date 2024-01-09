import src.p300speller.p300_speller as spl
from src.p300speller.p300_speller import ConfigParams, P300Window, MainWindow
from tkinter import Tk, Toplevel
from tkinter import Tk

log_file = "asda.txt"

spl.main(log_file)

#
# root = Tk()
# main_window = MainWindow(root, log_file)
# p300_window = main_window.open_p300_window()
# p300_window.start()
# while True:
#     root.update_idletasks()
#     root.update()



