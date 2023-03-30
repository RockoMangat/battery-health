# Get values from other scripts:
import pandas as pd

from U_av_charge_CC3 import charge_data
from U_av_discharge_CC2 import discharge_data
from incrementalcapacity6v2 import ica_data

# select dataset: 0, 1 or 2
datasetno = 0

# ------------------ Get the av charge voltage ------------------ #

result = charge_data(datasetno)

av_volt_charge = result[0]
charge_time_normalised = result[1]
volt_fixedtime = result[2]
cycles1 = result[3]
print('hello')

# ------------------ Get the ICA data ------------------ #
#

result2 = ica_data(datasetno)

maxica = result2[0]
cycles2 = result2[1]
print('hello')

#
# # # ------------------ Get the av discharge voltage ------------------ #


result3 = discharge_data(datasetno)

av_volt_discharge = result3[0]
cycles3 = result3[1]
print('hello')


# ------------------ Make dataframe ------------------ #
df1 = pd.DataFrame(list(zip(av_volt_charge, charge_time_normalised, volt_fixedtime)), index=cycles1)

df2 = pd.DataFrame(list(zip(maxica)), index=cycles2)

df3 = pd.DataFrame(list(zip(av_volt_discharge)), index=cycles3)


frames = [df1, df2, df3]
combinedframe = pd.concat(frames)

print('hello')
