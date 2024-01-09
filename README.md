# P300 speller for space project of Computational Psychiatry Lab from BGU and eeg sense
This is a p300 speller which made for the eeg sense head set

## Installation

The instrucitons assume you are installing on linux and have pipenv and python3.

1. Install dependencies
```shell
pipenv --python=python3
pipenv install --skip-lock
```
2. Update the config file in the p300speller folder and the consts.py 
3. Start the eeg measurement with sensi
4. Run the speller with
```
pipenv shell
python python src/p300_proccessing.py
```
