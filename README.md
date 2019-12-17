
# Repository for hand sensor data analysis
### Written by Ariel Morrison, University of Colorado/Cooperative Institute for Research in Environmental Sciences, ariel dot morrison at colorado dot edu
Reference for controlled setting: Morrison, A.L. et al. (2019), Quantifying student engagement in learning about climate change using galvanic hand sensors in a controlled educational setting. Climatic Change, https://doi.org/10.1007/s10584-019-02576-6.

*Description: This code performs a simple continuous decomposition analysis on electrodermal activity data downloaded from an Empatica E4 sensor.*


**To RUN:**

**Do steps 1-2 the FIRST time you run the code:**

1) Install clean virtual environment to run code:

`python3 -m pip install --user virtualenv`

Note: If your pip package is out of date, update the pip package when prompted.


2) Create virtual environment to download code requirements and run code:

`python3 -m venv env`

Note: If you get an error message "Error returned non-zero exit status 1," run this command without pip:

`python3 -m venv env --without-pip`



**If you've already installed a virtual environment in your working directory, start here:**

3) Activate your virtual environment:

`source env/bin/activate`


4) Install requirements.txt (contains all required packages) using pip:

Example command:

`pip install -r requirements.txt`

Note: If a package needs to be uninstalled before the requirements can be installed, use `conda uninstall $package`


5) Run the script with user inputs:

`python formattingSensorData.py $working_dir $timing_xcel $sheetname $beri_exists $beri_files $FS $delta $pref_dpi $separate_baseline $continuous_baseline $grade_files $grades_exist`


Example commands:

- Running with BERI protocol, continuous baseline, and grades
`python formattingSensorData.py "/Users/amorrison/Projects/hand_sensor_test/1060data" ATOC1060TimingComponents.xlsx total_timing_with_baseline True "beri_files" 4 0.25 800 False True "ATOC-1060-2018_Grades.xlsx" True`
- Running without BERI protocol, continuous baseline, without grades
`python formattingSensorData.py "/Users/amorrison/Projects/hand_sensor_test/1060data" ATOC1060TimingComponents.xlsx total_timing_with_baseline False "no_beri" 4 0.25 800 False True "no grades" False`
- Running with BERI protocol, entire semester baseline, and grades
`python formattingSensorData.py "/Users/jkay/Documents/jenkay/jek_research/ATOC1060_EducationResearch/hand_sensor_test-master/1060data" ATOC1060TimingComponents.xlsx total_timing True "beri_files" 4 0.25 800 False False "ATOC-1060-2018_Grades.xlsx" True`

**User-defined inputs:**
1. Working directory, where all downloaded zip archives are stored and where all output will be saved: $working_dir  --  e.g., "/Users/amorrison/Projects/hand_sensor_test/empaticadata" (put it in quotes)
2. Spreadsheet with component timing, including file extension: $timing_xcel -- e.g., study_timing.xlsx, study_timing.xls (no quotes, needs the file extension)
3. Sheet name in $timing_xcel: $sheetname -- e.g., exp_session (no quotes)
4. Are you using BERI observations? True or false: $beri_exists -- e.g., True (**NOTE**: only True or False are acceptable inputs)
5. BERI protocol observations directory: $beri_files -- e.g., "beri_files" (put it in quotes)
6. Sampling frequency per second: $Fs  --  e.g., 4 (4 default for E4 sensors, integer, units = samples per second)
7. Time interval of recording: $delta  --  e.g., 0.25 (0.25 default for E4 sensors, float, units = samples recorded every 0.25 seconds)
8. Preferred dpi (resolution) for saved .pdf figures: $pref_dpi -- e.g., 500, 1000, 2000 (900 is a good readable resolution)
9. Separate baseline recording? Baselines are separate if they are read in from a different file and then applied to one or more student records. True or False: $separate_baseline -- e.g., True (**NOTE**: only True or False are acceptable inputs. If baselines are recorded separately, must be stored in a subdirectory of the working directory called "calibration")
10. Continuous baseline recording? Baselines are continuous if they are part of the same skin conductance record to which they are being compared. For example, if the first 3 minutes of the skin conductance record are the 'baseline,' then choose True for the continuous baseline. True or False: $continuous_baseline -- e.g., False
11. Are you using grades in your analysis? True or false: $ grades_exist -- e.g., True (**NOTE**: only True or False are acceptable inputs)
12. Spreadsheet where grades are stored: $grade_files -- e.g., "ENV1000_grades.xlsx" (put it in quotes)


**Files saved to output directory:**

7 figures:
- total skin conductance from hand sensors (line)
- phasic component of skin conductance (line)
- tonic component of skin conductance (line)

*The 3 above figures are only made for one student at a time - i.e., there is no averaging. They are snapshots of one sensor's data.*

- bar chart of mean percent difference between activity and baseline skin conductance
- bar chart of mean percent difference between activity and baseline skin conductance, outliers removed
- bar chart of median percent difference between activity and baseline skin conductance
- histogram of mean percent difference between activity and baseline skin conductance

*The above figures are averages across all students.*


4 .csv files with statistics:
- for each activity: mean and median percent difference between activity and baseline skin conductance, std. deviation and std. error for mean percent difference, total time (in seconds) spent on each activity
- mean/standard deviation/standard error of skin conductance values for all sensors, all activities
- mean/standard deviation/standard error of number of engaged/disengaged students during class activities, based on BERI protocol
- grade breakdown, separated by STEM/non-STEM and gender (if grades are used)



# Common sources of error:
- Incorrect working directory. working_dir must be the directory where all input data are stored.
- Omitting user inputs. All 11 of the user-defined inputs must be included when calling the script.
- Incorrect user inputs. All Boolean inputs can only be True/False.
- Timing format is incorrect. When recording the activity timing in a spreadsheet, the format must always be YYYYMMDDHHMMSS. If the day and month columns are switched, the month column may end up out of range (e.g., if the date is September 28 and the month/day columns are switched then there will be an error because there are not 28 months).
