# Get values from other scripts:
datasetno = 0

# ------------------ Get the av charge voltage ------------------ #
#
# from U_av_charge_CC3 import charge_data
# # select dataset: 0, 1 or 2
# result = charge_data(datasetno)
#
#
# av_volt_charge = result[0]
# charge_time_normalised = result[1]
# volt_fixedtime = result[2]
# cycles = result[3]
# print('hello')

# ------------------ Get the av discharge voltage ------------------ #
#
# from U_av_discharge_CC2 import discharge_data
#
# result2 = discharge_data(datasetno)
#
#
# av_volt_discharge = result2[0]
# cycles2 = result2[1]
# print('hello')


# ------------------ Get the ICA data ------------------ #
#
from incrementalcapacity6 import ica_data

result3 = ica_data(datasetno)

maxica = result3[0]
cycles3 = result3[1]
print('hello')

