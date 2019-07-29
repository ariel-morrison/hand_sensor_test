
# "Test" repository for simple hand sensor data analysis
### Written by Ariel Morrison, University of Colorado/Cooperative Institute for Research in Environmental Sciences, ariel dot morrison at colorado dot edu

*Description: This code performs a simple continuous decomposition analysis on electrodermal activity data downloaded from an Empatica E4 sensor.*

**User-defined inputs:**
1. Working directory: $working_dir  --  e.g., "/Users/amorrison/Projects/hand_sensor_test/empaticadata" (put it in quotes) - this is where all downloaded zip archives are stored and where all output will be saved
2. Spreadsheet with component timing, including file extension: $timing_xcel -- e.g., study_timing.xlsx, study_timing.xls (no quotes, needs the file extension)
3. Sheet name in $timing_xcel: $sheetname -- e.g., exp_session (no quotes)
4. Sampling frequency per second: $Fs  --  e.g., 4 (4 default for E4 sensors, integer, units = samples per second)
5. Time interval of recording: $delta  --  e.g., 0.25 (0.25 default for E4 sensors, float, units = samples recorded every 0.25 seconds)
6. Preferred format for saved figures: $pref_format -- e.g., png, eps, pdf
7. Preferred dpi (resolution) for saved figures: $pref_dpi -- e.g., 500, 1000, 2000 (1000 is a good readable resolution for png files)


**To RUN:**

Do steps 1-2 the FIRST time you run the code:

1) Install clean virtual environment to run code:

`python3 -m pip install --user virtualenv`


2) Create virutal environment:

`python3 -m venv env`


If you've already installed a virtual environment, start here:

3) Activate the virtual environment:

`source env/bin/activate`


4) Install requirements.txt using pip:

Example command:

`pip install -r requirements.txt`


5) Run the script

`python formattingSensorData.py $working_dir $timing_xcel $sheetname $FS $delta $pref_format $pref_dpi`


Example commands:

`python formattingSensorData.py "/Users/amorrison/Projects/hand_sensor_test/empaticadata" test_timing.xlsx exp_session 4 0.25 png 1000`


`python formattingSensorData.py "/Users/jkay/Documents/jenkay/jek_research/handsensors/hand_sensor_test/empaticadata" test_timing.xlsx exp_session 4 0.25 eps 2000`


**Output saved to working directory:**

5 figures:
- total skin conductance from hand sensors (line)
- phasic component of skin conductance (line)
- tonic component of skin conductance (line)

*The 3 above figures are only made for one sensor at a time - i.e., there is no averaging. They are snapshots of one sensor's data.*

- mean percent difference between activity and baseline skin conductance (bar/column), std. error
- median percent difference between activity and baseline skin conductance (bar/column), std. error

*The above figures are averages across all sensors.*


2 .csv files with statistics:
- mean and median percent difference between activity and baseline skin conductance for all activities, including std. deviation/std. error
- average skin conductance values for all sensors, all activities - baseline with stddev, activity with stddev
