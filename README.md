

# "Test" repository for simple hand sensor data analysis
### Written by Ariel Morrison, University of Colorado/Cooperative Institute for Research in Environmental Sciences, ariel dot morrison at colorado dot edu

**Description: This code performs a simple continuous decomposition analysis on electrodermal activity data downloaded from an Empatica E4 sensor.**

User-defined inputs:
1. Working directory: $working_dir  --  e.g., "/Users/amorrison/Projects/hand_sensor_test/empaticadata" (put it in quotes) - this is where all downloaded zip archives are stored and where all output will be saved
2. Sampling frequency per second: $Fs  --  e.g., 4 (4 default for E4 sensors, integer, units samples per second)
3. Time interval of recording: $delta  --  e.g., 0.25 (0.25 default for E4 sensors, float, units samples recorded every 0.25 seconds)
4. Length of baseline in minutes: $min_baseline  --  e.g., 3 (integer in minutes)
5. Preferred format for saved figures: $pref_format -- e.g., png, eps, pdf
6. Preferred dpi (resolution) for saved figures: $pref_dpi -- e.g., 500, 1000, 2000 (1000 is a good readable resolution for png files)


To RUN:

1) Install requirements.txt using pip:

Example command:

`pip install -r requirements.txt`

2) Run the script

`python formattingSensorData.py $working_dir $FS $delta $min_baseline $pref_format $pref_dpi`


Example command:

`python formattingSensorData.py "/Users/amorrison/Projects/hand_sensor_test/empaticadata" 4 0.25 3 png 1000`


`python formattingSensorData.py "/Users/jkay/Documents/jenkay/jek_research/handsensors/hand_sensor_test/empaticadata" 4 0.25 3 eps 2000`

3) Output:

3 figures:
- total skin conductance from hand sensors
- phasic component of skin conductance
- mean percent difference between activity and baseline skin conductance

1 csv file with statistics:
- mean and median percent difference between activity and baseline skin conductance for all activities
