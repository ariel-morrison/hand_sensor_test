# "Test" repository for simple hand sensor data analysis

This code performs a simple continuous decomposition analysis on electrodermal activity data downloaded from an Empatica E4 sensor.

User-defined inputs:
1. Working directory: mydir
2. Sampling frequency per second: Fs --> for E4 sensors, 4 samples per second
3. Time interval of recording: delta --> for E4 sensors, samples recorded every 0.25s
4. Length of baseline in minutes (to calculate how many timesteps to take): min_baseline

