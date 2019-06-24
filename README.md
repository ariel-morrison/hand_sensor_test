

# "Test" repository for simple hand sensor data analysis
### Written by Ariel Morrison, University of Colorado/Cooperative Institute for Research in Environmental Sciences, ariel dot morrison at colorado dot edu

**Description: This code performs a simple continuous decomposition analysis on electrodermal activity data downloaded from an Empatica E4 sensor.**

User-defined inputs:
1. Working directory: $working_dir  --  e.g., "/Users/amorrison/Projects/hand_sensor_test/empaticadata" (put it in quotes)
2. Sampling frequency per second: $Fs  --  e.g., 4 (4 default for E4 sensors, integer, units samples per second)
3. Time interval of recording: $delta  --  e.g., 0.25 (0.25 default for E4 sensors, float, units samples recorded every 0.25 seconds)
4. Length of baseline in minutes: $min_baseline  --  e.g., 3 (integer in minutes)


To RUN:

1) Install requirements.txt using pip:

Example command:
`pip install -r requirements.txt`

2) Run the script

`python formattingSensorData.py $working_dir $FS $delta $min_baseline`

Example command:

`python formattingSensorData.py "/Users/amorrison/Projects/hand_sensor_test/empaticadata" 4 0.25 3`
`python formattingSensorData.py "/Users/jkay/Documents/jenkay/jek_research/handsensors/hand_sensor_test/empaticadata" 4 0.25 3`
