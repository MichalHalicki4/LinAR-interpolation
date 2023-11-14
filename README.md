# LinAR interpolation method
This repository contains a python implementation of the **LinAR** interpolation approach, which allows to interpolate gaps in time series with the joint use of autoregression and linear interpolation.

For a detailed description of the **LinAR** interpolation method and its use, see:

Niedzielski, T., Halicki, M. 2023. Improving Linear Interpolation of Missing Hydrological Data by Applying Integrated Autoregressive Models. Water Resources Management. <href>https://doi.org/10.1007/s11269-023-03625-7</href>


# Content
This repository consits of 3 python files and an example time series.

SCRIPTS:
- `LinAR_functions.py` - includes all functions neccesary to conduct **LinAR** interpolation.
- `LinAR_executor.py` - includes the main interpolation function, which uses functions declared in `LinAR_functions.py`.
- `LinAR_example.py` - provides an example of conducting the **LinAR** interpolation on a time series. 

DATA:
- Here we used the water level time series from the Barmah gauge on the Murray River, provided by the Bureu of Meteorology, Australia, under the CC BY 4.0 license. These measurements can be downloaded from http://www.bom.gov.au/waterdata/. However, we created additional gaps in these measurements to better present the potential of the **LinAR** method.

# How to use the **LinAR interpolation** method?

To use the python implementation of this method, the following packages have to be installed:

- matplotlib,
- numpy,
- pandas,
- scipy,
- statsmodels.

The **LinAR** method can be used by running the `LinAR_executor.py` file. There are several neccesary variables which have to be provided, so the script runs properly:

- _filepath_ - The full filename (path + file name) of the .csv sheet with data to be interpolated.
- _column_dates_ - The header of the column containing the dates of measurements.
- _column_id_ - The header of the column with observations to be interpolated.
- _separator_ - The separator of the .csv sheet. The default is set to ';'.
- _learn_len_ - The size of the train data for the autoregression (selected in a moving window, before each gap). The default is set to 100.
- _max_lags_ - The maximum number of autoregressive lags included in the model. The default is set to 10.
- _max_linear_ - The maximum gap size to be linearly interpolated. The default value is set to 72.
- _max_linar_ - The maximum gap size to be linearly interpolated. The recommended value is set to 12. Further explanations can be found in **Niedzielski and Halicki (2023)**.
- _sig_adf_ - The signigicance level for the ADF test. The default is set to 0.05.
- _sig_ft_ - The signigicance level for the F test. The default is set to 0.05.
- _number_of_diffs_ - The number of differencings allowed to obtain stationary data. The default value is set to 2.
- _output_file_  - The full output file name (path + file name).

These variables can be declared interactively, by following the instructions displayed in the console. It is also possible to declare these variables manually, within the script - in this situation the _interactive_input_ variable (Line 10) should be changed to 'False', and the other variables should be defined below, in lines 25-36. An example use of the **LinAR** method has been presented in the `LinAR_example.py` script. 


# LICENSE

When using this method please cite the following article:
Niedzielski, T., and Halicki, M. Can integrated autoregressive modelling improve linear interpolation of missing hydrological data? [Submitted].


MIT License

Copyright (c) 2023 MichalHalicki4

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
