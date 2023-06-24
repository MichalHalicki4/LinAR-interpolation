import pandas as pd
from LinAR_functions import interpolate_linar
import matplotlib.pyplot as plt

"""
DECLARE THE VARIABLES NECESSARY TO RUN THE LINAR INTERPOLATION.
YOU CAN DO IT IN THE CONSOLE - JUST RUN THE SCRIPT AND FOLLOW THE INSTRUCTIONS.
YOU CAN ALSO DO IT WITHIN THE SCRIPT - CHANGE INTERACTIVE_INPUT TO FALSE AND DECLARE NECESSARY VARIABLES MANUALLY
"""
interactive_input = True
if interactive_input:
    filepath = input('INSERT YOUR FILE NAME: ')
    column_dates = input('INSERT THE HEADER OF THE DATES COLUMN: ')
    column_id = input(' INSERT THE HEADER OF THE COLUMN WITH OBSERVATIONS TO BE INTERPOLATED: ')
    separator = input("INSERT THE DATA SEPARATOR. DEFAULT = ';' ")
    default_vals = input('DO YOU WANT TO CHANGE DEFAULT VARIABLES? PLEASE TYPE Y or N for yes or no, respectively')
    if default_vals not in ['y', 'Y', 'yes', 'YES', 'Yes']:
        learn_len = 100
        max_lags = 10
        max_linear = 72
        max_linar = 14
        sig_adf = 0.05
        sig_ft = 0.05
        number_of_diffs = 2
    else:
        learn_len = int(input('INSERT THE SIZE OF THE TRAIN DATA FOR THE AUTOREGRESSION. DEFAULT = 100: '))
        max_lags = int(input('INSERT THE MAXIMUM NUMBER OF AUTOREGRESSIVE LAGS INCLUDED IN THE MODEL. DEFAULT = 10: '))
        max_linear = int(input('INSERT THE MAXIMUM GAP SIZE TO BE LINEARLY INTERPOLATED. DEFAULT = 72: '))
        max_linar = int(input('INSERT THE MAXIMUM GAP SIZE TO BE INTERPOLATED WITH THE LINAR METHOD. RECOMMENDED = 14: '))
        sig_adf = float(input('INSERT THE SIGNIFICANCE LEVEL FOR THE ADF TEST. DEFAULT = 0.05: '))
        sig_ft = float(input('INSERT THE SIGNIFICANCE LEVEL FOR THE F TEST. DEFAULT = 0.05: '))
        number_of_diffs = int(input('INSERT NUMBER OF DIFFERENCINGS ALLOWED IN THE WHILE LOOP. DEFAULT = 2: '))

    output_file = input('INSERT YOUR OUTPUT FILE NAME: ')
else:
    filepath = 'example_data/gauge_409215_with_artificial_gaps.csv'  # INSERT YOUR FILE NAME.
    column_dates = '#Timestamp'  # INSERT THE HEADER OF THE DATES COLUMN.
    column_id = 'Value'  # INSERT THE HEADER OF THE COLUMN WITH OBSERVATIONS TO BE INTERPOLATED.
    separator = ';'  # DEFAULT DATA SEPARATOR = ';', IF IT IS DIFFERENT (E.G. ',' OR ' ') PLEASE UPDATE THIS VARIABLE.
    learn_len = 100  # INSERT THE SIZE OF THE TRAIN DATA FOR THE AUTOREGRESSION. DEFAULT = 100.
    max_lags = 15  # INSERT THE MAXIMUM NUMBER OF AUTOREGRESSIVE LAGS INCLUDED IN THE MODEL. DEFAULT = 10.
    max_linear = 72  # INSERT THE MAXIMUM GAP SIZE TO BE LINEARLY INTERPOLATED. DEFAULT = 72.
    max_linar = 12  # INSERT THE MAXIMUM GAP SIZE TO BE INTERPOLATED WITH THE LINAR METHOD. RECOMMENDED = 12.
    sig_adf = 0.1  # THE SIGNIFICANCE LEVEL FOR THE ADF TEST. DEFAULT = 0.05.
    sig_ft = 0.01  # THE SIGNIFICANCE LEVEL FOR THE F TEST. DEFAULT = 0.05.
    number_of_diffs = 3  # NUMBER OF DIFFERENCINGS ALLOWED IN THE WHILE LOOP. DEFAULT = 2.
    output_file = 'interpolated.csv'  # INSERT YOUR OUTPUT FILE NAME

# READ THE DATA AND RUN THE INTERPOLATION
df = pd.read_csv(filepath, sep=separator, index_col=column_dates, parse_dates=True)
itpd = interpolate_linar(df, column_id, learn_len, max_lags, max_linear, max_linar, sig_adf, sig_ft, number_of_diffs)

# SAVE THE INTERPOLATED TIME SERIES TO A .CSV FILE.
itpd.to_csv(output_file, sep=';')

# PLOT THE RESULTS
fig, ax = plt.subplots()
ax.plot(itpd, color='red', label='interpolated')
ax.plot(df[column_id], color='grey', linewidth=1.5, label='true observations')
ax.legend()
plt.show()
