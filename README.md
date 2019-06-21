# "Test" repository for simple hand sensor data analysis

This code performs a simple continuous decomposition analysis on electrodermal activity data downloaded from an Empatica E4 sensor.

User-defined inputs:
1. Working directory: working_dir
2. Sampling frequency per second: Fs --> for E4 sensors, 4 samples per second (integer)
3. Time interval of recording: delta --> for E4 sensors, samples recorded every 0.25s (float)
4. Length of baseline in minutes (to calculate how many timesteps to take): min_baseline (integer)

To RUN install requirements.txt using pip and:
`python formattingSensorData.py "working_dir" Fs delta min_baseline`

Example usage:
`python formattingSensorData.py "/Users/amorrison/Projects/hand_sensor_test/empaticadata" 4 0.25 3`
