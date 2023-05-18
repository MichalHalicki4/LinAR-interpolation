import pandas as pd
import numpy as np
# from datetime import timedelta, datetime
from main import linar_interpolate
import random
import matplotlib.pyplot as plt

# df = pd.read_csv('https://query.data.world/s/r4hhh7xr42g3l22zisculw3ftutqa6?dws=00000', sep=';')
# df = df.loc[df['PRODUCT_ID'] == 2]
# df['SURVEY_DATE'] = pd.to_datetime(df['SURVEY_DATE'])
# df.set_index('SURVEY_DATE', inplace=True)
# column_id = 'PRICE'
# df_resampled = df[column_id].resample('W').mean()
# df_resampled_copy = df_resampled.copy()
# dates = df_resampled.index.to_pydatetime()
# stp = dates[1] - dates[0]
# for index, row in df_resampled_copy.items():
#     if np.isnan(row):
#         for i in range(random.randint(1, 10)):
#             df_resampled.loc[index-i*stp] = np.nan
#             df_resampled.loc[index+i*stp] = np.nan


# READ DATA AND INDICATE THE COLUMN NAME, WHICH SHOULD BE INTERPOLATED
# df = pd.read_csv('/Users/michalhalicki/Documents/nauka/projekty/Sonata-BIS/program_sonata/althydro'
#                  '/interpolation_github/github/archive/archive/DailyDelhiClimateTrain.csv', sep=',',
#                  index_col='date', parse_dates=True)
# column_id = 'meantemp'

df = pd.read_csv('/Users/michalhalicki/Documents/nauka/projekty/Sonata-BIS/program_sonata/althydro'
                 '/interpolation_github/github/example_data/csv.w00074.20230518092046.409215.csv', sep=',',
                 skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8], index_col='#Timestamp', parse_dates=True)
df = df.loc['2017-01-01': '2020-01-01']
column_id = 'Value'

# CREATING A DATAFRAME COPY AND RANDOM GAPS, SINCE THE ORIGINAL DATASET DOES NOT CONTAIN ANY
df_copy = df.copy()
random.seed(10)
random_indexes = random.sample(range(320, 1000), 40)
for indx in random_indexes:
    for i in range(random.randint(1, 6)):
        df.iloc[indx-i] = np.nan
        df.iloc[indx+i] = np.nan


# DECLARE THE VARIABLES NECESSARY TO RUN THE LINAR INTERPOLATION
learn_len = 100
max_lags = 10
max_linear, max_linar = 72, 12
sig_adf, sig_ft = 0.05, 0.01
number_of_diffs = 2

# RUNNING THE INTERPOLATION
itpd = linar_interpolate(df, column_id, learn_len, max_lags, max_linear, max_linar, sig_adf, sig_ft, number_of_diffs)

# PLOTTING THE RESULTS
fig, ax = plt.subplots()
ax.plot(itpd, color='red', label='interpolated')
ax.plot(df_copy[column_id], color='grey', linewidth=1.5, label='true observations')
ax.set_ylabel('Water level [m]')
ax.set_xlabel('Time')
ax.legend()
plt.show()
