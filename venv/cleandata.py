from scipy.io import loadmat, whosmat
import numpy as np
import datetime


# import data from other file
from attempt3 import test_dict_reconstructed


# creating a function to make separate dictionaries
def build_dictionaries(mess):

# creating three separate dictionaries for discharge, charge and impedance
    discharge, charge, impedance = {}, {}, {}

    for z in mess:
        print('z:', z)

# iterating using the enumerate function which gives two loop variables: 1. count and 2. value of item, at current iteration
        for i, v in enumerate(z):
            # count is 0, and value is B0005:

            #print(i, v)

            print('count is: ', i)
            print('value is: ', v)


            step = v[0]
            print('step: ', step)

    # for discharge data values:
            if step == 'discharge':
                print(step)
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

                discharge[str(i)]["voltage_battery"] = data[0][0][0][0].tolist()
                discharge[str(i)]["current_battery"] = data[0][0][1][0].tolist()
                discharge[str(i)]["temp_battery"] = data[0][0][2][0].tolist()
                discharge[str(i)]["current_load"] = data[0][0][3][0].tolist()
                discharge[str(i)]["voltage_load"] = data[0][0][4][0].tolist()
                discharge[str(i)]["time"] = data[0][0][5][0].tolist()


            print(discharge)

    return discharge, charge, impedance

build_dictionaries(test_dict_reconstructed)

