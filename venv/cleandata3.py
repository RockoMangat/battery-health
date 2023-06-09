import scipy.io as sio
from scipy.io import loadmat, whosmat
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import os

# load .mat file
all_data = sio.loadmat('/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat')

# test = all_data[0]

# print(test)

# only extract data needed from file:
mat_data = {}
for key in all_data.keys():
    if not key.startswith('__'):
        mat_data[key] = all_data[key]


def build_dictionaries(mess):
    discharge, charge, impedance = {}, {}, {}

    for i, element in enumerate(mess):

        print('i:', i)
        print('element:', element)

        # checks if the step will be charge, discharge, impedance
        step = element[0][0]

        print('step:', step)

        if step == 'discharge':
            discharge[str(i)] = {}
            discharge[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1) * 1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)

            discharge[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            test1 = data[0]
            test2 = data[0][0]
            test3 = data[0][0][0]
            test4 = data[0][0][0][0]


            discharge[str(i)]["voltage_battery"] = data[0][0][0][0].tolist()
            discharge[str(i)]["current_battery"] = data[0][0][1][0].tolist()
            discharge[str(i)]["temp_battery"] = data[0][0][2][0].tolist()
            discharge[str(i)]["current_load"] = data[0][0][3][0].tolist()
            discharge[str(i)]["voltage_load"] = data[0][0][4][0].tolist()
            discharge[str(i)]["time"] = data[0][0][5][0].tolist()
            discharge[str(i)]["capacity"] = data[0][0][6][0].tolist()


        if step == 'charge':
            charge[str(i)] = {}
            charge[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1) * 1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)

            charge[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            charge[str(i)]["voltage_battery"] = data[0][0][0][0].tolist()
            charge[str(i)]["current_battery"] = data[0][0][1][0].tolist()
            charge[str(i)]["temp_battery"] = data[0][0][2][0].tolist()
            charge[str(i)]["current_load"] = data[0][0][3][0].tolist()
            charge[str(i)]["voltage_load"] = data[0][0][4][0].tolist()
            charge[str(i)]["time"] = data[0][0][5][0].tolist()

        if step == 'impedance':
            impedance[str(i)] = {}
            impedance[str(i)]["amb_temp"] = str(element[1][0][0])
            year = int(element[2][0][0])
            month = int(element[2][0][1])
            day = int(element[2][0][2])
            hour = int(element[2][0][3])
            minute = int(element[2][0][4])
            second = int(element[2][0][5])
            millisecond = int((second % 1) * 1000)
            date_time = datetime.datetime(year, month, day, hour, minute, second, millisecond)

            impedance[str(i)]["date_time"] = date_time.strftime("%d %b %Y, %H:%M:%S")

            data = element[3]

            impedance[str(i)]["sense_current"] = {}
            impedance[str(i)]["battery_current"] = {}
            impedance[str(i)]["current_ratio"] = {}
            impedance[str(i)]["battery_impedance"] = {}
            impedance[str(i)]["rectified_impedance"] = {}

            impedance[str(i)]["sense_current"]["real"] = np.real(data[0][0][0][0]).tolist()
            impedance[str(i)]["sense_current"]["imag"] = np.imag(data[0][0][0][0]).tolist()

            impedance[str(i)]["battery_current"]["real"] = np.real(data[0][0][1][0]).tolist()
            impedance[str(i)]["battery_current"]["imag"] = np.imag(data[0][0][1][0]).tolist()

            impedance[str(i)]["current_ratio"]["real"] = np.real(data[0][0][2][0]).tolist()
            impedance[str(i)]["current_ratio"]["imag"] = np.imag(data[0][0][2][0]).tolist()

            impedance[str(i)]["battery_impedance"]["real"] = np.real(data[0][0][3]).tolist()
            impedance[str(i)]["battery_impedance"]["imag"] = np.imag(data[0][0][3]).tolist()

            impedance[str(i)]["rectified_impedance"]["real"] = np.real(data[0][0][4]).tolist()
            impedance[str(i)]["rectified_impedance"]["imag"] = np.imag(data[0][0][4]).tolist()

            impedance[str(i)]["re"] = float(data[0][0][5][0][0])
            impedance[str(i)]["rct"] = float(data[0][0][6][0][0])

    # print(discharge, charge, impedance)

    return discharge, charge, impedance


folder = '/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4'
filenames = [f for f in os.listdir(folder) if f.endswith('.mat')]

for filename in filenames:
    name = filename.split('.mat')[0]
    print(name)
    # loading file
    struct = loadmat(folder + '/' + filename)
    # selecting one of the battery datasets
    mess = struct[name][0][0][0][0]
    # print('struct', struct)

    # print(mess)

    mess2 = struct[name][0][0][0]

    mess3 = struct[name][0][0]
    print(type(mess3))

    mess4 = struct[name][0]

    mess5 = struct[name]

    mess_1 = struct[name][0][0][0][0][0]

    mess_2 = struct[name][0][0][0][0][0][0]

    mess_3 = struct[name][0][0][0][0][0][0][0]

    mess_4 = struct[name][0][0][0][0][0][0][0][0]

    # thus mess is the right one to use - it iterates over charge, discharge and impedance, any less/further deep in and get wrong values

    discharge, charge, impedance = build_dictionaries(mess)

# print(discharge)
