README

Requirements:
Python version 3 must be installed on the computer running this model

Data Preparation:
Save unknown mapped_brands as a .csv file named 'input_data' to the directory named 'data' with a column labeled 'merchant_string' (which must be filled with non-NA values), a column labeled 'mcc' (which may contain NA values if they are label 'NA'), and a column labeled 'network'.
See the file 'input_data_sample_format.csv' in the data directory for more clarity.

Run:
If any of the following are not installed please run the following commands in the command line.
pip install wheels
pip install pandas
pip install numpy
pip install scipy

Run the 'run.py' to generate labels for the 'input_data.csv' file. If you run this in a python ide, then once models are saved into memory as python objects the code will run faster.


Output:
In the 'output_data_labeled.csv' file will be written to the data directory.
This file will contain the input data with 3 additional columns.
	1. Confidence - model confidence level for each observation - this is not analogous to the probability a prediction is correct
	2. All_Predict - mapped_brand predictions for all inputed data
	3. Predict - mapped_brand predictions for inputed data above the recommended model confidence level