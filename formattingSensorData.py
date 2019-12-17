# Script to unzip downloaded EDA files from Empatica website, analyze skin conductance data, plot data, and save output as csv

import os
import sys
import zipfile
import shutil
import pandas as pd
import numpy as np
import cvxopt as cv
import cvxopt.solvers
import statistics
from scipy import stats
from statistics import mean
import datetime
import pytz
from datetime import datetime
import time
import csv
import pylab as pl
import plotly
import plotly.graph_objs as go


def cvxEDA(obs_EDA, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2, solver=None, options={'reltol': 1e-9, 'show_progress': False}):
    # default options = same as Ledalab
    """CVXEDA Convex optimization approach to electrodermal activity processing
    Arguments:
       obs_EDA: observed skin conductance signal (we recommend normalizing it: obs_EDA = zscore(obs_EDA))
       delta: sampling interval (in seconds) of obs_EDA
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options - 'reltol' = relative accuracy, 'abstol' = absolute accuracy, 'feastol' = tolerance for feasibility conditions

    Returns (see paper for details):
       phasic = phasic component
       p: sparse SMNA driver of phasic component
       tonic: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)

       from Greco et al. (2016). cvxEDA: A Convex Optimization Approach to Electrodermal Activity Processing,
       IEEE Transactions on Biomedical Engineering, 63(4): 797-804.
    """

    n = len(obs_EDA)
    obs_EDA = cv.matrix(obs_EDA)

    # bateman ARMA model
    a1 = 1./min(tau1, tau0) # a1 > a0
    a0 = 1./max(tau1, tau0)
    ar = np.array([(a1*delta + 2.) * (a0*delta + 2.), 2.*a1*a0*delta**2 - 8.,
        (a1*delta - 2.) * (a0*delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))
    M = cv.spmatrix(np.tile(ma, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1.,delta_knot_s), np.arange(delta_knot_s, 0., -1.)] # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl)//2), (len(spl)+1)//2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl),1))
    p = np.tile(spl, (nB,1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n+1.)/n])
    nC = C.size[1]

    # solve the problem
    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m,n: cv.spmatrix([],[],[],(m,n))
        G = cv.sparse([[-A,z(2,n),M,z(nB+2,n)],[z(n+2,nC),C,z(nB+2,nC)],
                    [z(n,1),-1,1,z(n+nB+2,1)],[z(2*n+2,1),-1,1,z(nB,1)],
                    [z(n+2,nB),B,z(2,nB),cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n,1),.5,.5,obs_EDA,.5,.5,z(nB,1)])
        c = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T,z(nC,1),1,gamma,z(nB,1)])
        res = cv.solvers.conelp(c, G, h, dims={'l':n,'q':[n+2,nB+2],'s':[]})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt*M, Ct*M, Bt*M], [Mt*C, Ct*C, Bt*C],
                    [Mt*B, Ct*B, Bt*B+gamma*cv.spmatrix(1.0, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T - Mt*obs_EDA,  -(Ct*obs_EDA), -(Bt*obs_EDA)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n,len(f))),
                            cv.matrix(0., (n,1)), solver=solver)
        obj = res['primal objective'] + .5 * (obs_EDA.T * obs_EDA)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n+nC]
    tonic = B*l + C*d
    q = res['x'][:n]
    p = A * q
    phasic = M * q
    e = obs_EDA - phasic - tonic

    # return [np.array(a).ravel() for a in (phasic, p, tonic, l, d, e, obj)]
    return [np.array(a).ravel() for a in (phasic, tonic, e)]



def extract_zip_format_filenames(working_dir):
    """
    Input: working directory (working_dir) where all data are downloaded from Empatica website.

    Goal: move all data from downloaded zip archives into working directory

    What it does: Searches working_dir and all subdirectories for .zip archives, unzips all zipped archives,
    extracts the sensor name and date (first 10 digits of file name) as 'zipfile_name,' prints
    the sensor number as a check, and then pulls out the EDA, HR, and tags.csv files from each archive.
    All .csv files are extracted to the working_dir and renamed with the sensor name/date.
    E.g., 1523940183_A0108B_EDA.csv, 1523940183_A0108B_HR.csv...
    All EDA, HR, tag files are appended to lists when they're extracted to the working_dir, so the
    rest of the functions read data out of the lists and keep files in the same order.
    """

    zip_list = []
    EDA_list = []
    HR_list = []
    tag_list = []

    for dirpath, dirnames, filenames in os.walk(working_dir): # goes through every file in working_dir and all subdirectories
        dirnames[:] = [d for d in dirnames if not d.startswith('calibration')] # skips the calibration directory if there's a separate baseline
    # name of current dirctory, directories inside current dir, and files inside current dir
        for filename in filenames:
            # for the current dir, for all filenames in that dir...
            if '.zip' in filename:
                # is string.zip in string filename?
                path_to_zip_file = os.path.join(dirpath, filename)
                zip_list.append(path_to_zip_file)
                zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
                zipfile_name = os.path.splitext(os.path.basename(path_to_zip_file))[0]

                # check if the zip archive has already been unzipped
                # zipfile_name is sensor number
                if not os.path.exists(zipfile_name):
                    os.mkdir(zipfile_name)
                    zip_ref.extractall(os.path.join(working_dir, zipfile_name))
                    zip_ref.close()

                sensorNum = path_to_zip_file[-21:-4]

                # Check current working directory.
                working_sub_dir = os.path.join(working_dir, sensorNum)

                eda_filepath = os.path.join(working_sub_dir, 'EDA.csv')
                if os.path.isfile(eda_filepath): # check if an EDA.csv file exists in the folder
                    eda_filename = working_dir + '/' + str(sensorNum) + '_EDA.csv'
                    os.rename(eda_filepath, eda_filename)
                    EDA_list.append(eda_filename)

                # if os.path.isfile(os.path.join(working_sub_dir, 'HR.csv')): # check if a HR.csv file exists in the folder
                #     hr_filename = working_dir + '/' + str(sensorNum) + '_HR.csv'
                #     os.rename(working_sub_dir + '/' + 'HR.csv', hr_filename)
                #     HR_list.append(hr_filename)
                #
                # if os.path.isfile(os.path.join(working_sub_dir, 'tags.csv')): # check if a tags.csv file exists in the folder
                #     tag_filename = working_dir + '/' + str(sensorNum) + '_tags.csv'
                #     os.rename(working_sub_dir + '/' + 'tags.csv', tag_filename)
                #     tag_list.append(tag_filename)

                shutil.rmtree(working_sub_dir)

    return zip_list, EDA_list, HR_list, tag_list



def get_activity_timing(working_dir, timing_xcel, sheetname, EDA_data_df, EDA_data_df2, beri_exists):
    """
    Input: working directory (working_dir) where all data are downloaded from Empatica website;
            spreadsheet (timing_xcel) where all component timing is recorded (see example);
            sheet name in spreadsheet (sheetname) where all component timing is recorded;
            skin conductance dataframe (EDA_data_df)

    Goal: Find data within specific date/time ranges

    What it does: Opens the spreadsheet where all component timing is recorded, reads through
    each row of the starting time ('datetime_start') and creates a YYYYMMDDHHMMSS timestamp,
    reads through each row of the ending time ('datetime_end') and creates a YYYYMMDDHHMMSS timestamp,
    then reads through every timestamp of the skin conductance dataframe to find the values that fall
    within the start and end time of each component. Also counts the total number of seconds spent on
    each class activity.
    """

    os.chdir(working_dir)

    # lambda = defines anonymous functions that can only produce one line of output but still require varied inputs
    excel_timing = os.path.join(working_dir, str(timing_xcel))

    xcel = pd.read_excel(excel_timing, sheet_name = str(sheetname))
    xcel['datetime_start'] = xcel.apply(lambda row: datetime.strptime(str(row['Year Start']) + \
                                                                                           str(row['Month Start']).zfill(2) + \
                                                                                           str(row['Day Start']).zfill(2) + \
                                                                                           str(row['Hour Start']).zfill(2) + \
                                                                                           str(row['Minute Start']).zfill(2) + \
                                                                                           str(row['Second Start']).zfill(2), "%Y%m%d%H%M%S"), axis=1)
    xcel['datetime_end'] = xcel.apply(lambda row: datetime.strptime(str(row['Year End']) + \
                                                                                         str(row['Month End']).zfill(2) + \
                                                                                         str(row['Day End']).zfill(2) + \
                                                                                         str(row['Hour End']).zfill(2) + \
                                                                                         str(row['Minute End']).zfill(2) + \
                                                                                         str(row['Second End']).zfill(2), "%Y%m%d%H%M%S"), axis=1)

    x_out = xcel.apply(lambda row : EDA_data_df[(EDA_data_df['timestamp']>=row['datetime_start'])&(EDA_data_df['timestamp']<row['datetime_end'])].assign(activity=row['Activity']), axis=1)
    activity_mean = pd.concat(list(x_out)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].mean()
    activity_stddev = pd.concat(list(x_out)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].std()
    activity_stderr = pd.concat(list(x_out)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].sem()

    if beri_exists == True:
        x_out_beri = xcel.apply(lambda row : EDA_data_df2[(EDA_data_df2['timestamp']>=row['datetime_start'])&(EDA_data_df2['timestamp']<row['datetime_end'])].assign(activity=row['Activity']), axis=1)
        activity_mean_beri = pd.concat(list(x_out_beri)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].mean()


    # select all activities except for exams
    xcel_activities = xcel
    xcel_activities = xcel_activities.reset_index().set_index('Activity')
    xcel_activities['datetime_start'] = pd.to_datetime(xcel_activities['datetime_start'])
    xcel_activities['datetime_end'] = pd.to_datetime(xcel_activities['datetime_end'])
    # idx = xcel_activities.index
    # print(idx)
    # if idx.contains('Exam') == True:
    #     xcel_activities = xcel_activities[~idx.contains("Exam")]
    #
    # print("xcel_activities:")
    # print(xcel_activities)
    EDA_data_df['timestamp'] = pd.to_datetime(EDA_data_df['timestamp'])

    act_start = xcel_activities['datetime_start']
    act_end = xcel_activities['datetime_end']


    if beri_exists == False:
        mask = []
        for idx in range(0,len(act_start)): # take only the skin conductance that falls between the start/stop time of study activities
            mask1 = EDA_data_df[(EDA_data_df['timestamp'] > act_start.iloc[idx]) & (EDA_data_df['timestamp'] <= act_end.iloc[idx])]
            mask.append(mask1)

        mask = pd.concat(mask, axis=0)
        del mask1

        baseline_activities = mask.groupby(['sensor_ids']).mean() # baseline for each student = average skin conductance during activities

    if beri_exists == True:
        mask = []
        for idx in range(0,len(act_start)):
            mask1 = EDA_data_df2[(EDA_data_df2['timestamp'] > act_start.iloc[idx]) & (EDA_data_df2['timestamp'] <= act_end.iloc[idx])]
            mask.append(mask1)

        mask = pd.concat(mask, axis=0)
        del mask1

        baseline_activities_beri = mask.groupby(['sensor_ids']).mean()


    # to get the total time spent on each class activity
    xcel['total_time'] = pd.to_datetime(xcel['datetime_end'], infer_datetime_format=True) - pd.to_datetime(xcel['datetime_start'], infer_datetime_format=True)
    total_time = xcel[['Activity','total_time']]
    total_time = total_time.set_index('Activity')
    #total_time = total_time.drop(['Total'], axis=0)
    if (total_time.index == 'Baseline').any() == True:
        total_time = total_time.drop(['Baseline'], axis=0)
    total_time = total_time.groupby(['Activity'])['total_time'].sum()

    # to get the total number of seconds spent on each class activity
    xcel['total_time_seconds'] = xcel['total_time'].dt.total_seconds()
    total_time_seconds = xcel[['Activity', 'total_time_seconds']]
    total_time_seconds = total_time_seconds.set_index('Activity')
    #total_time_seconds = total_time_seconds.drop(['Total'], axis=0)
    if (total_time_seconds.index == 'Baseline').any() == True:
        total_time_seconds = total_time_seconds.drop(['Baseline'], axis=0)
    total_time_seconds = total_time_seconds.groupby(['Activity'])['total_time_seconds'].sum()

    if beri_exists == True:
        return activity_mean, activity_mean_beri, activity_stddev, activity_stderr, total_time, total_time_seconds, EDA_data_df, baseline_activities_beri
    else:
        return activity_mean, activity_stddev, activity_stderr, total_time, total_time_seconds, EDA_data_df, baseline_activities


def reduce_function(row, data_reduce, student_overview):
    if not isinstance(row.name, pd.Timestamp):
        pass
    seat_num = int(row.index[0].split('-')[1])
    out = np.NaN
    if row[(row.index[row.index.str.contains('-E|-W|-L', regex=True)])].any():
        out = True
    elif row[(row.index[row.index.str.contains('-D|-U|-S', regex=True)])].any():
        out = False
    if len(student_overview.loc[str(row.name.normalize())][student_overview.loc[str(row.name.normalize())] == seat_num].index) > 0:
        data_reduce.at[str(row.name), student_overview.loc[str(row.name.normalize())][student_overview.loc[str(row.name.normalize())] == seat_num].index[0]] = out


def get_beri_protocol(working_dir, beri_files, beri_exists):
    """
    Input: working directory (working_dir) where all data are downloaded from Empatica website;
            spreadsheet (timing_beri) where BERI protocol observations are recorded (see example)

    Goal: Find how many students exhibited engaged/disengaged behaviors

    What it does: Opens the folder where all BERI observations are recorded, sums the number of
    engaged/disengaged students during each type of activity, then normalizes it by the number of
    instances of that activity
    """
    os.chdir(working_dir)

    beri_df = []
    beri_data = []

    if beri_exists == True :
        student_overview = pd.read_excel(os.path.join(working_dir, "StudentDataOverview.xlsx"))
        student_overview = student_overview.set_index('Sensor').T
        student_overview.index = pd.to_datetime(student_overview.index).normalize()

        beri_dir = os.path.join(working_dir, 'beri_files')
        os.chdir(beri_dir)


        for dirpath, dirnames, filenames in os.walk(beri_dir):
            for filename in filenames:
                if 'Our Changing Environment' in filename:
                    path_to_beri_file = os.path.join(dirpath, filename)
                    data = pd.read_csv(filename, parse_dates=[['class_date','time']])
                    data = data[data.columns.drop(list(data.filter(regex=('id|observer|instructor|class_subject_code|class_number|value|Instructor_Activity|Notes'))))] # drop columns that don't include student behaviors
                    data = data.sort_values("class_date_time")
                    data = data.set_index('class_date_time')

                    prefixes = [c.split('-')[1] if '-' in c else c for c in data.columns]
                    prefixes = list(dict.fromkeys(prefixes))
                    for p in prefixes:
                        p = int(p)

                    grouper = [next(p for p in prefixes if (p == (c.split('-')[1]))) for c in data.columns]
                    data_grouped = data.groupby(grouper, axis=1)

                    data_reduce = pd.DataFrame(index=data.index)
                    data_grouped.apply(lambda df: df.apply(reduce_function, axis=1, data_reduce=data_reduce, student_overview=student_overview))
                    data_reduce = data_reduce.resample('250L', label='right', closed='right').nearest().ffill()
                    beri_df.append(data_reduce)

                    ########
                    data = pd.read_csv(filename, parse_dates=[['class_date','time']])
                    beri_data.append(data)

        beri_df = pd.concat(beri_df, sort=False)
        beri_data = pd.concat(beri_data, sort=False)
        beri_data['total_eng'] = beri_data[(beri_data.columns[beri_data.columns.str.contains('-E')] | beri_data.columns[beri_data.columns.str.contains('-L')] | beri_data.columns[beri_data.columns.str.contains('-W')])].sum(axis=1)
        beri_data['total_diseng'] = beri_data[(beri_data.columns[beri_data.columns.str.contains('-D')] | beri_data.columns[beri_data.columns.str.contains('-U')] | beri_data.columns[beri_data.columns.str.contains('-S')])].sum(axis=1)
        beri_data = beri_data.drop(['id', 'class_number'], axis=1).sort_values("class_date_time")

    return beri_df, beri_data



def get_grades(working_dir, grade_files, EDA_by_sensor):
    """
    Input: working directory (working_dir) where all data are downloaded from Empatica website;
            spreadsheet (grade_files) where grades and students' sensor numbers are recorded

    Goal: Compare students' engagement levels with their grades

    What it does: Opens the grade spreadsheet, reads the sensor number and associated grade
    """

    os.chdir(working_dir)
    grades_all = pd.read_excel(os.path.join(working_dir, grade_files))

    grades = []
    # Sensor Count = sensor number
    grades = grades_all.loc[grades_all['Sensor ID'] != 0]
    grades = grades[['Class Level','STEM/non-STEM [STEM major=1, non-STEM major=2, undeclared=3]','Gender [male=1, female=2, other=3]','Midterm #1','Midterm #2','Final Exam','Homework','Final Course Grade','Sensor ID']]
    grades = grades.rename(columns={'STEM/non-STEM [STEM major=1, non-STEM major=2, undeclared=3]': 'STEM=1, non-STEM=2, undec=3', 'Final Course Grade':'Final Grade', 'Sensor ID':'sensor_ids'})

    stem = grades.loc[grades['STEM=1, non-STEM=2, undec=3'] == 1]
    nonstem = grades.loc[grades['STEM=1, non-STEM=2, undec=3'] == 2]
    undec = grades.loc[grades['STEM=1, non-STEM=2, undec=3'] == 3]
    female = grades.loc[grades['Gender [male=1, female=2, other=3]'] == 2]
    male = grades.loc[grades['Gender [male=1, female=2, other=3]'] == 1]


    index = [('STEM', 'Midterm #1'), ('STEM', 'Midterm #2'), ('STEM', 'Final Exam'), ('STEM', 'Final Grade'),
             ('Non-STEM', 'Midterm #1'), ('Non-STEM', 'Midterm #2'), ('Non-STEM', 'Final Exam'), ('Non-STEM','Final Grade'),
             ('Undeclared', 'Midterm #1'), ('Undeclared', 'Midterm #2'), ('Undeclared', 'Final Exam'), ('Undeclared', 'Final Grade'),
             ('Female', 'Midterm #1'), ('Female', 'Midterm #2'), ('Female', 'Final Exam'), ('Female', 'Final Grade'),
             ('Male', 'Midterm #1'), ('Male', 'Midterm #2'), ('Male', 'Final Exam'), ('Male', 'Final Grade')]
    numbers = [stem['Midterm #1'].mean(), stem['Midterm #2'].mean(), stem['Final Exam'].mean(), stem['Final Grade'].mean(),
                   nonstem['Midterm #1'].mean(), nonstem['Midterm #2'].mean(), nonstem['Final Exam'].mean(), nonstem['Final Grade'].mean(),
                   undec['Midterm #1'].mean(), undec['Midterm #2'].mean(), undec['Final Exam'].mean(), undec['Final Grade'].mean(),
                   female['Midterm #1'].mean(), female['Midterm #2'].mean(), female['Final Exam'].mean(), female['Final Grade'].mean(),
                   male['Midterm #1'].mean(), male['Midterm #2'].mean(), male['Final Exam'].mean(), male['Final Grade'].mean()]

    sep_grades = pd.Series(numbers, index=index)
    index = pd.MultiIndex.from_tuples(index)
    sep_grades = sep_grades.reindex(index).round(2)

    sep_grades_df = pd.DataFrame({'Avg. Grade': sep_grades,
                                 'Std. Dev': [stem['Midterm #1'].std(), stem['Midterm #2'].std(), stem['Final Exam'].std(), stem['Final Grade'].std(),
                                             nonstem['Midterm #1'].std(), nonstem['Midterm #2'].std(), nonstem['Final Exam'].std(), nonstem['Final Grade'].std(),
                                             undec['Midterm #1'].std(), undec['Midterm #2'].std(), undec['Final Exam'].std(), undec['Final Grade'].std(),
                                             female['Midterm #1'].std(), female['Midterm #2'].std(), female['Final Exam'].std(), female['Final Grade'].std(),
                                             male['Midterm #1'].std(), male['Midterm #2'].std(), male['Final Exam'].std(), male['Final Grade'].std()],
                                 'Std. Err': [stem['Midterm #1'].sem(), stem['Midterm #2'].sem(), stem['Final Exam'].sem(), stem['Final Grade'].sem(),
                                             nonstem['Midterm #1'].sem(), nonstem['Midterm #2'].sem(), nonstem['Final Exam'].sem(), nonstem['Final Grade'].sem(),
                                             undec['Midterm #1'].sem(), undec['Midterm #2'].sem(), undec['Final Exam'].sem(), undec['Final Grade'].sem(),
                                             female['Midterm #1'].sem(), female['Midterm #2'].sem(), female['Final Exam'].sem(), female['Final Grade'].sem(),
                                             male['Midterm #1'].sem(), male['Midterm #2'].sem(), male['Final Exam'].sem(), male['Final Grade'].sem()]}).round(2)

    sep_grades_df.to_csv("separated grades.csv")

    clicker_q = pd.read_excel(os.path.join(working_dir, "ATOC1060_Fall2018_Clickers_IRBresearch.xlsx"), usecols="A,E:F")
    clicker_q_df = clicker_q.reset_index().groupby("Lecture Date").mean()
    clicker_q_df['avg%correct'] = clicker_q_df[['%correct', '%correct 2nd time']].mean(axis=1)
    clicker_q_df.to_csv("clicker questions.csv")


    print("Completed grades")
    print(" ")

    return sep_grades_df, clicker_q_df, grades


def plot_results(Fs, pref_dpi, EDA_data_df, EDA_data_df2, output_dir, separate_baseline, continuous_baseline, beri_exists, EDA_by_sensor, grades_exist):
#def plot_results(obs_EDA, phasic, tonic, Fs, pref_dpi, EDA_data_df, output_dir, separate_baseline, continuous_baseline, beri_exists, EDA_by_sensor, grades_exist):
    """
    Input: for plotting an individual's data - skin conductance dataframe (EDA_data_df), phasic/tonic components
    Sampling frequency per second (Fs), preferred figure resolution (pref_dpi)
    For plotting average data, what type of baseline (separate, continuous, neither), whether the BERI beri_protocol
    was used, and the functions that process skin conductance data.

    Goal: To produce figures and save them to output directory

    What it does: Plots line graphs of an individual's total, phasic, and tonic components of skin conductance
    against minutes. Calculates percent difference in mean skin conductance between an activity and baseline, plots
    bar graph for mean/median percent difference for each activity. Plots histogram of mean percent difference.
    """

#     timing = pl.arange(1., len(obs_EDA) + 1.) / (60 * Fs) # minutes = divide by 240 = 60 seconds * 4 records/sec
#
# # plotting total conductance (phasic + tonic + noise)
#     fig1, ax = pl.subplots( nrows=1, ncols=1 )
#     pl.plot(timing, obs_EDA, color = 'r')
#     pl.xlim(0, max(timing) + 1)
#     pl.ylabel('Skin conductance - total (\u03bcS)')
#     pl.xlabel('Time (min)')
#     fig1.savefig(os.path.join(output_dir, 'total_conductance.png'), dpi = pref_dpi)
#     pl.close(fig1)
#
# # plotting phasic component of skin conductance
#     ylim_top = max(phasic)
#     fig2, ax = pl.subplots( nrows=1, ncols=1 )
#     pl.plot(timing, phasic, color = 'b')
#     pl.xlim(0, max(timing) + 1)
#     pl.ylabel('Skin conductance - phasic component (\u03bcS)')
#     pl.xlabel('Time (min)')
#     fig2.savefig(os.path.join(output_dir, 'phasic_component.png'), dpi = pref_dpi)
#     pl.close(fig2)
#
# # plotting tonic component of skin conductance
#     ylim_top = max(tonic)
#     fig3, ax = pl.subplots( nrows=1, ncols=1 )
#     pl.plot(timing, tonic, color = 'g')
#     pl.xlim(-1, max(timing) + 1)
#     pl.ylabel('Skin conductance - tonic component (\u03bcS)')
#     pl.xlabel('Time (min)')
#     fig3.savefig(os.path.join(output_dir, 'tonic_component.png'), dpi = pref_dpi)
#     pl.close(fig3)


    if grades_exist == True:
        sep_grades_df, clicker_q_df, grades = get_grades(working_dir, grade_files, EDA_by_sensor)

    def outliers_to_nan(activity):
        threshold = 3
        percent_diffs = (activity["skin_conduct_means"] - activity["skin_conduct_baseline"]) / activity["skin_conduct_baseline"]
        mean = percent_diffs.mean()
        std = percent_diffs.std()

        z_score = ((percent_diffs - mean)/std).abs()
        activity['outlier'] = z_score > threshold
        return activity

    def calculate_percent_diff(row):
        return ((row['skin_conduct_means'] - row['skin_conduct_baseline'])/row['skin_conduct_baseline'])*100

    # get timing and EDA for each activity
    if beri_exists == True:
        activity_mean, activity_mean_beri, activity_stddev, activity_stderr, total_time, total_time_seconds, EDA_data_df, baseline_activities_beri = get_activity_timing(working_dir, timing_xcel, sheetname, EDA_data_df, EDA_data_df2, beri_exists)
        activity_mean_beri = activity_mean_beri.reset_index()
        activity_mean_beri = activity_mean_beri.rename(columns={'level_0': 'file_name'})
        print("activity_mean_beri:")
        print(activity_mean_beri)
        beri_df, beri_data = get_beri_protocol(working_dir, beri_files, beri_exists)
        beri_data.to_csv("beri_obs_total.csv")

    else:
        activity_mean, activity_stddev, activity_stderr, total_time, total_time_seconds, EDA_data_df, baseline_activities = get_activity_timing(working_dir, timing_xcel, sheetname, EDA_data_df, EDA_data_df2, beri_exists)

    activity_mean = activity_mean.reset_index().rename(columns={'level_0': 'file_name'})

    # changes to calibration directory if user input was "true" for separate baselines
    if separate_baseline == True :
        calibration_dir = os.path.join(working_dir, 'calibration')
        os.chdir(calibration_dir)

        zip_list = []
        baseline_df = pd.DataFrame()

        for dirpath, dirnames, filenames in os.walk(calibration_dir):
            for filename in filenames:
                if '.zip' in filename:
                    # is string.zip in string filename?
                    path_to_zip_file = os.path.join(dirpath, filename)
                    zip_list.append(path_to_zip_file)
                    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
                    zipfile_name = os.path.splitext(os.path.basename(path_to_zip_file))[0]

                    if not os.path.exists(zipfile_name):
                        os.mkdir(zipfile_name)
                        zip_ref.extractall(os.path.join(calibration_dir, zipfile_name))
                        zip_ref.close()

                    sensorNum = path_to_zip_file[-21:-4]
                    sensorNum_no_ts = path_to_zip_file[-10:-4]

                    calibration_sub_dir = os.path.join(calibration_dir, sensorNum)

                    baseline_filepath = os.path.join(calibration_sub_dir, 'EDA.csv')
                    if os.path.isfile(baseline_filepath): # check if an EDA.csv file exists in the folder
                        baseline_filename = os.path.join(calibration_dir, str(sensorNum) + '_EDA.csv')
                        # reads in baseline data records from each student
                        temp_df = pd.read_csv(baseline_filepath, header=2, names=['skin_conduct_baseline'])
                        temp_df = temp_df[1200:-1200]
                        os.rename(baseline_filepath, baseline_filename)
                        temp_df['file_name_no_ts'] = str(sensorNum_no_ts)
                        baseline_df = baseline_df.append(temp_df)

                    shutil.rmtree(calibration_sub_dir)

            # finds mean baseline for each student, puts all baselines in a dataframe and sorts by sensor number
            baselines = baseline_df.groupby(['file_name_no_ts'])['skin_conduct_baseline'].mean().reset_index()

            for i in range(0,len(baselines)):
                if (baseline_df.groupby(['file_name_no_ts'])['skin_conduct_baseline'].max().reset_index())['skin_conduct_baseline'][i] > 2.5*(baseline_df.groupby(['file_name_no_ts'])['skin_conduct_baseline'].min().reset_index())['skin_conduct_baseline'][i]:
                    baselines['skin_conduct_baseline'][i] = np.nan

        """
        remove baseline from dataframe, if it existed as part of the continuous data record; rename columns;
        convert the sensor ID to a string; split the sensor ID string at the underscore to separate timestamp
        from actual sensor ID number; merge the dataframe containing sensor ID, activity mean skin conductance,
        and baselines for each student
        """
        activity_mean_no_bl = activity_mean[activity_mean['activity'] != "Baseline"].rename(columns = {"skin_conduct":"skin_conduct_means"})
        activity_mean_no_bl["file_name_no_ts"] = activity_mean_no_bl['file_name'].astype(str)
        activity_mean_no_bl["file_name_no_ts"] = activity_mean_no_bl["file_name_no_ts"].str.split('_').str[1]

        activity_mean_merged = activity_mean_no_bl.merge(baselines, on = ["file_name_no_ts"])
        activity_mean_merged = activity_mean_merged.rename(columns = {'file_name_no_ts':'sensor_ids'})

        if beri_exists == True:
            activity_mean_no_bl_beri = activity_mean_beri[activity_mean_beri['activity'] != "Baseline"]
            activity_mean_no_bl_beri = activity_mean_no_bl_beri.rename(columns = {"skin_conduct":"skin_conduct_means"})
            activity_mean_no_bl_beri["file_name_no_ts"] = activity_mean_no_bl_beri['file_name'].astype(str)
            activity_mean_no_bl_beri["file_name_no_ts"] = activity_mean_no_bl_beri["file_name_no_ts"].str.split('_').str[1]
            activity_mean_merged_beri = activity_mean_no_bl_beri.merge(baselines, on = ["file_name_no_ts"])
            activity_mean_merged_beri = activity_mean_merged_beri.rename(columns = {"file_name_no_ts":"sensor_ids"})
            print("activity_mean_merged_beri:")
            print(activity_mean_merged_beri)
            print(" ")

        print("Separate baseline")
        print(" ")


# If baseline method = continous (part of class/study):
    elif continuous_baseline == True :
        print("Continuous baseline")
        print(" ")
        if beri_exists == False:
            print("activity_mean:")
            print(activity_mean)
            print(" ")
            baselines = activity_mean[activity_mean['activity'] == "Baseline"][["file_name", "skin_conduct"]]
            baselines = baselines.rename(columns = {"skin_conduct":"skin_conduct_baseline"})
            print("baselines:")
            print(baselines)
            print(" ")
            activity_mean_no_bl = activity_mean[activity_mean['activity'] != "Baseline"]
            activity_mean_no_bl["file_name_no_ts"] = activity_mean_no_bl['file_name'].astype(str)
            activity_mean_no_bl["file_name_no_ts"] = activity_mean_no_bl["file_name_no_ts"].str.split('_').str[1]
            activity_mean_no_bl = activity_mean_no_bl.rename(columns = {"skin_conduct":"skin_conduct_means"})
            activity_mean_merged = activity_mean_no_bl.merge(baselines, on = ["file_name"])
            activity_mean_merged = activity_mean_merged.rename(columns = {"file_name_no_ts":"sensor_ids"})

            new_column = activity_mean_merged.groupby(['sensor_ids']).apply(calculate_percent_diff)
            activity_mean_merged['% diff'] = new_column.reset_index(level=0, drop=True).rename(columns = {"file_name_no_ts":"sensor_ids"})
            activity_mean_merged = activity_mean_merged.groupby(['activity']).apply(outliers_to_nan)
            activity_mean_merged = activity_mean_merged[~activity_mean_merged['outlier']]

            percent_diff_means_no_outliers = activity_mean_merged[~activity_mean_merged['outlier']].groupby(['activity']).mean()
            percent_diff_means_no_outliers = percent_diff_means_no_outliers['% diff']

            # mean/median percent difference between baseline and activity
            activity_mean_merged = activity_mean_merged.drop(['file_name', 'outlier'], axis=1)
            percent_diff_stderr_no_outliers = activity_mean_merged.groupby(['activity']).sem()
            percent_diff_stderr_no_outliers = percent_diff_stderr_no_outliers['% diff']
            percent_diff_stddev_no_outliers = activity_mean_merged.groupby(['activity']).std()
            percent_diff_stddev_no_outliers = percent_diff_stddev_no_outliers['% diff']
            percent_diff_medians_no_outliers = activity_mean_merged.groupby(['activity']).median()
            percent_diff_medians_no_outliers = percent_diff_medians_no_outliers['% diff']

            total_percent_diff = activity_mean_merged.groupby(['sensor_ids']).mean()
            total_percent_diff = total_percent_diff['% diff']

            if grades_exist == True:
                grades_merged = grades.merge(total_percent_diff, left_on='sensor_ids', right_on='sensor_ids').replace([np.inf, -np.inf], np.nan).dropna()


        if beri_exists == True:
            baselines = activity_mean_beri[activity_mean_beri['activity'] == "Baseline"][["file_name", "skin_conduct"]]
            baselines = baselines.rename(columns = {"skin_conduct":"skin_conduct_baseline"})
            print("baselines:")
            print(baselines)
            print(" ")
            activity_mean_no_bl_beri = activity_mean_beri[activity_mean_beri['activity'] != "Baseline"]
            activity_mean_no_bl_beri["file_name_no_ts"] = activity_mean_no_bl_beri['file_name'].astype(str)
            activity_mean_no_bl_beri["file_name_no_ts"] = activity_mean_no_bl_beri["file_name_no_ts"].str.split('_').str[1]
            activity_mean_no_bl_beri = activity_mean_no_bl_beri.rename(columns = {"skin_conduct":"skin_conduct_means"})
            activity_mean_merged_beri = activity_mean_no_bl_beri.merge(baselines, on = ["file_name"])
            activity_mean_merged_beri = activity_mean_merged_beri.rename(columns = {"file_name_no_ts":"sensor_ids"})

            new_column = activity_mean_merged_beri.groupby(['sensor_ids']).apply(calculate_percent_diff)
            activity_mean_merged_beri['% diff'] = new_column.reset_index(level=0, drop=True).rename(columns = {"file_name_no_ts":"sensor_ids"})

            activity_mean_merged_beri = activity_mean_merged_beri.groupby(['activity']).apply(outliers_to_nan)
            activity_mean_merged_beri = activity_mean_merged_beri[~activity_mean_merged_beri['outlier']]

            percent_diff_means_no_outliers_beri = activity_mean_merged_beri[~activity_mean_merged_beri['outlier']].groupby(['activity']).mean()
            percent_diff_means_no_outliers_beri = percent_diff_means_no_outliers_beri['% diff']

            activity_mean_merged_beri = activity_mean_merged_beri.drop(['file_name', 'outlier'], axis=1)

            percent_diff_stderr_no_outliers_beri = activity_mean_merged_beri.groupby(['activity']).sem()
            percent_diff_stderr_no_outliers_beri = percent_diff_stderr_no_outliers_beri['% diff']
            percent_diff_stddev_no_outliers_beri = activity_mean_merged_beri.groupby(['activity']).std()
            percent_diff_stddev_no_outliers_beri = percent_diff_stddev_no_outliers_beri['% diff']
            percent_diff_medians_no_outliers_beri = activity_mean_merged_beri.groupby(['activity']).median()
            percent_diff_medians_no_outliers_beri = percent_diff_medians_no_outliers_beri['% diff']

            total_percent_diff = activity_mean_merged_beri.groupby(['sensor_ids']).mean()
            total_percent_diff = total_percent_diff['% diff']

            if grades_exist == True:
                grades_merged_beri = grades.merge(total_percent_diff, left_on='sensor_ids', right_on='sensor_ids').replace([np.inf, -np.inf], np.nan).dropna()


# If baseline method = entire record (averaged over entire semester, day, etc):
    else:
        print("Entire semester baseline")
        print(" ")

        if beri_exists == False:
            baselines = baseline_activities['skin_conduct']
            activity_mean_no_bl = activity_mean[activity_mean['activity'] != "Baseline"].rename(columns = {"skin_conduct":"skin_conduct_means"})
            activity_mean_no_bl["file_name_no_ts"] = activity_mean_no_bl['file_name'].astype(str)
            activity_mean_no_bl["file_name_no_ts"] = activity_mean_no_bl["file_name_no_ts"].str.split('_').str[1]

            activity_mean_merged = activity_mean_no_bl.rename(columns = {"file_name_no_ts":"sensor_ids"})
            activity_mean_merged = activity_mean_merged.merge(baselines.to_frame(), on = ['sensor_ids']).rename(columns = {"skin_conduct" : "skin_conduct_baseline"})

            new_column = activity_mean_merged.groupby(['sensor_ids']).apply(calculate_percent_diff)
            activity_mean_merged['% diff'] = new_column.reset_index(level=0, drop=True).rename(columns = {"file_name_no_ts":"sensor_ids"})
            activity_mean_merged = activity_mean_merged.groupby(['activity']).apply(outliers_to_nan)
            activity_mean_merged = activity_mean_merged[~activity_mean_merged['outlier']]

            percent_diff_means_no_outliers = activity_mean_merged[~activity_mean_merged['outlier']].groupby(['activity']).mean()
            percent_diff_means_no_outliers = percent_diff_means_no_outliers['% diff']

            # mean/median percent difference between baseline and activity
            activity_mean_merged = activity_mean_merged.drop(['file_name', 'outlier'], axis=1)
            percent_diff_stderr_no_outliers = activity_mean_merged.groupby(['activity']).sem()
            percent_diff_stderr_no_outliers = percent_diff_stderr_no_outliers['% diff']
            percent_diff_stddev_no_outliers = activity_mean_merged.groupby(['activity']).std()
            percent_diff_stddev_no_outliers = percent_diff_stddev_no_outliers['% diff']
            percent_diff_medians_no_outliers = activity_mean_merged.groupby(['activity']).median()
            percent_diff_medians_no_outliers = percent_diff_medians_no_outliers['% diff']

            total_percent_diff = activity_mean_merged.groupby(['sensor_ids']).mean()
            total_percent_diff = total_percent_diff['% diff']

            if grades_exist == True:
                grades_merged = grades.merge(total_percent_diff, left_on='sensor_ids', right_on='sensor_ids').replace([np.inf, -np.inf], np.nan).dropna()


        if beri_exists == True:
            baselines = baseline_activities_beri['skin_conduct']
            activity_mean_no_bl_beri = activity_mean_beri[activity_mean_beri['activity'] != "Baseline"]
            activity_mean_no_bl_beri = activity_mean_no_bl_beri.rename(columns = {"skin_conduct":"skin_conduct_means"})
            activity_mean_no_bl_beri["file_name_no_ts"] = activity_mean_no_bl_beri['file_name'].astype(str)
            activity_mean_no_bl_beri["file_name_no_ts"] = activity_mean_no_bl_beri["file_name_no_ts"].str.split('_').str[1]

            activity_mean_merged_beri = activity_mean_no_bl_beri.rename(columns = {"file_name_no_ts":"sensor_ids"})
            activity_mean_merged_beri = activity_mean_merged_beri.merge(baselines.to_frame(), on = ['sensor_ids']).rename(columns = {"skin_conduct" : "skin_conduct_baseline"})
            print("activity_mean_merged_beri:")
            print(activity_mean_merged_beri)
            print(" ")
            new_column = activity_mean_merged_beri.groupby(['sensor_ids']).apply(calculate_percent_diff)
            activity_mean_merged_beri['% diff'] = new_column.reset_index(level=0, drop=True).rename(columns = {"file_name_no_ts":"sensor_ids"})

            activity_mean_merged_beri = activity_mean_merged_beri.groupby(['activity']).apply(outliers_to_nan)
            activity_mean_merged_beri = activity_mean_merged_beri[~activity_mean_merged_beri['outlier']]

            percent_diff_means_no_outliers_beri = activity_mean_merged_beri[~activity_mean_merged_beri['outlier']].groupby(['activity']).mean()
            percent_diff_means_no_outliers_beri = percent_diff_means_no_outliers_beri['% diff']

            activity_mean_merged_beri = activity_mean_merged_beri.drop(['file_name', 'outlier'], axis=1)

            percent_diff_stderr_no_outliers_beri = activity_mean_merged_beri.groupby(['activity']).sem()
            percent_diff_stderr_no_outliers_beri = percent_diff_stderr_no_outliers_beri['% diff']
            percent_diff_stddev_no_outliers_beri = activity_mean_merged_beri.groupby(['activity']).std()
            percent_diff_stddev_no_outliers_beri = percent_diff_stddev_no_outliers_beri['% diff']
            percent_diff_medians_no_outliers_beri = activity_mean_merged_beri.groupby(['activity']).median()
            percent_diff_medians_no_outliers_beri = percent_diff_medians_no_outliers_beri['% diff']

            total_percent_diff = activity_mean_merged_beri.groupby(['sensor_ids']).mean()
            total_percent_diff = total_percent_diff['% diff']

            if grades_exist == True:
                grades_merged_beri = grades.merge(total_percent_diff, left_on='sensor_ids', right_on='sensor_ids').replace([np.inf, -np.inf], np.nan).dropna()


    if grades_exist == True:
        fig1, ax = pl.subplots( nrows=1, ncols=1 )
        if beri_exists == True:
            pl.scatter(grades_merged_beri['Final Grade'], grades_merged_beri['% diff'], c = 'k', marker='o', s=13)
        else:
            pl.scatter(grades_merged['Final Grade'], grades_merged['% diff'], c = 'k', marker='o', s=13)
        pl.yticks(fontsize=10, fontweight='bold')
        pl.xticks(fontsize=10, fontweight='bold')
        pl.xlim(60,102)
        if beri_exists == True:
            pl.ylim(min(grades_merged_beri['% diff']-15), max(grades_merged_beri['% diff']+15))
        else:
            pl.ylim(min(grades_merged['% diff']-15), max(grades_merged['% diff']+15))
        pl.ylabel('Engagement relative to baseline (%)', fontweight='bold')
        pl.xlabel('Final Course Grade', fontweight='bold')
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.2)
        pl.tight_layout()
        if separate_baseline == True:
            fig1.savefig(os.path.join(output_dir,'final_grades_vs_conductance_separate_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True:
            fig1.savefig(os.path.join(output_dir,'final_grades_vs_conductance_continuous_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig1.savefig(os.path.join(output_dir,'final_grades_vs_conductance_entire_semester_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig1)

        fig2, ax = pl.subplots( nrows=1, ncols=1 )
        if beri_exists == True:
            pl.scatter(grades_merged_beri['Midterm #1'], grades_merged_beri['% diff'], c = 'r', marker='o', s=12)
        else:
            pl.scatter(grades_merged['Midterm #1'], grades_merged['% diff'], c = 'r', marker='o', s=12)
        pl.yticks(fontsize=9, fontweight='bold')
        pl.xticks(fontsize=9, fontweight='bold')
        pl.xlim(60,102)
        if beri_exists == True:
            pl.ylim(min(grades_merged_beri['% diff']-15), max(grades_merged_beri['% diff']+15))
        else:
            pl.ylim(min(grades_merged['% diff']-15), max(grades_merged['% diff']+15))
        pl.ylabel('Engagement relative to baseline (%)', fontweight='bold')
        pl.xlabel('Midterm #1 Grade', fontweight='bold')
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.2)
        pl.tight_layout()
        if separate_baseline == True:
            fig2.savefig(os.path.join(output_dir,'midterm1_vs_conductance_separate_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True:
            fig2.savefig(os.path.join(output_dir,'midterm1_vs_conductance_continuous_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig2.savefig(os.path.join(output_dir,'midterm1_vs_conductance_entire_semester_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig2)

        fig3, ax = pl.subplots( nrows=1, ncols=1 )
        if beri_exists == True:
            pl.scatter(grades_merged_beri['Midterm #2'], grades_merged_beri['% diff'], c = 'g', marker='o', s=12)
        else:
            pl.scatter(grades_merged['Midterm #2'], grades_merged['% diff'], c = 'g', marker='o', s=12)
        pl.yticks(fontsize=9, fontweight='bold')
        pl.xticks(fontsize=9, fontweight='bold')
        pl.xlim(60,102)
        if beri_exists == True:
            pl.ylim(min(grades_merged_beri['% diff']-15), max(grades_merged_beri['% diff']+15))
        else:
            pl.ylim(min(grades_merged['% diff']-15), max(grades_merged['% diff']+15))
        pl.ylabel('Engagement relative to baseline (%)', fontweight='bold')
        pl.xlabel('Midterm #2 Grade', fontweight='bold')
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.2)
        pl.tight_layout()
        if separate_baseline == True:
            fig3.savefig(os.path.join(output_dir,'midterm2_vs_conductance_separate_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True:
            fig3.savefig(os.path.join(output_dir,'midterm2_vs_conductance_continuous_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig3.savefig(os.path.join(output_dir,'midterm2_vs_conductance_entire_semester_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig3)

        fig4, ax = pl.subplots( nrows=1, ncols=1 )
        if beri_exists == True:
            pl.scatter(grades_merged_beri['Final Exam'], grades_merged_beri['% diff'], c = 'b', marker='o', s=13)
        else:
            pl.scatter(grades_merged['Final Exam'], grades_merged['% diff'], c = 'b', marker='o', s=13)
        pl.yticks(fontsize=10, fontweight='bold')
        pl.xticks(fontsize=10, fontweight='bold')
        pl.xlim(60,102)
        if beri_exists == True:
            pl.ylim(min(grades_merged_beri['% diff']-15), max(grades_merged_beri['% diff']+15))
        else:
            pl.ylim(min(grades_merged['% diff']-15), max(grades_merged['% diff']+15))
        pl.ylabel('Engagement relative to baseline (%)', fontweight='bold')
        pl.xlabel('Final Exam Grade', fontweight='bold')
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.2)
        pl.tight_layout()
        if separate_baseline == True:
            fig4.savefig(os.path.join(output_dir,'final_exam_vs_conductance_separate_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True:
            fig4.savefig(os.path.join(output_dir,'final_exam_vs_conductance_continuous_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig4.savefig(os.path.join(output_dir,'final_exam_vs_conductance_entire_semester_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig4)


    if beri_exists == False:
        statistics_output = percent_diff_means_no_outliers, percent_diff_stddev_no_outliers, percent_diff_stderr_no_outliers, \
                            percent_diff_medians_no_outliers, total_time, total_time_seconds
        percent_diff_means_idx = list(percent_diff_means_no_outliers.index)
        y_pos = {key: percent_diff_means_idx[key-1] for key in range(1, (len(percent_diff_means_idx)+1), 1)}
        keywords = y_pos.values()

        # mean percent difference, no outliers
        fig7, ax = pl.subplots( nrows=1, ncols=1 )
        print("percent_diff_means_no_outliers:")
        print(percent_diff_means_no_outliers)
        pl.bar(list(y_pos.keys()), percent_diff_means_no_outliers, yerr=percent_diff_stderr_no_outliers, error_kw=dict(lw=0.65, capsize=2, capthick=0.55), align='center', color=[0.62,0.07,0.41], alpha=1)
        pl.xticks(list(y_pos.keys()), list(y_pos.values()), rotation=90, fontsize=6)
        pl.ylim(min((percent_diff_means_no_outliers-percent_diff_stderr_no_outliers-10)), max(percent_diff_means_no_outliers+percent_diff_stderr_no_outliers+15))
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.22, left=0.12)
        pl.tight_layout()
        pl.ylabel('Mean skin conductance % difference w/o outliers\n(activity - baseline)', fontsize=6)
        pl.yticks(fontsize=6)
        if separate_baseline == True :
            fig7.savefig(os.path.join(output_dir, 'activity_means_no_outliers_separate_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True :
            fig7.savefig(os.path.join(output_dir, 'activity_means_no_outliers_continuous_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig7.savefig(os.path.join(output_dir, 'activity_means_no_outliers_entire_semester_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig7)


        # median percent difference, no outliers
        fig8, ax = pl.subplots( nrows=1, ncols=1 )
        pl.bar(list(y_pos.keys()), percent_diff_medians_no_outliers, align='center', color=[0.89,0.07,0.41], alpha=1)
        pl.xticks(list(y_pos.keys()), list(y_pos.values()), rotation=90, fontsize=6)
        pl.ylim(min(percent_diff_medians_no_outliers-5), max(percent_diff_medians_no_outliers+15))
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.22, left=0.12)
        pl.tight_layout()
        pl.ylabel('Median skin conductance % difference w/o outliers\n(activity - baseline)', fontsize=6)
        pl.yticks(fontsize=6)
        if separate_baseline == True :
            fig8.savefig(os.path.join(output_dir, 'activity_medians_no_outliers_separate_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True :
            fig8.savefig(os.path.join(output_dir, 'activity_medians_no_outliers_continuous_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig8.savefig(os.path.join(output_dir, 'activity_medians_no_outliers_entire_semester_BL.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig8)


        # histogram
        fig10, ax = pl.subplots( nrows=1, ncols=1 )
        pl.hist(percent_diff_means_no_outliers[np.isfinite(percent_diff_means_no_outliers)].values, bins=26, color=[0.85,0.33,0], align='mid', rwidth=0.92)
        pl.ylabel('Counts')
        pl.xlabel("Mean skin conductance % difference from baseline, no outliers")
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.22, left=0.12)
        pl.tight_layout()
        if separate_baseline == True :
            fig10.savefig(os.path.join(output_dir, 'activity_means_no_outliers_separate_BL_hist.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True :
            fig10.savefig(os.path.join(output_dir, 'activity_means_no_outliers_continuous_BL_hist.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig10.savefig(os.path.join(output_dir, 'activity_means_no_outliers_entire_semester_BL_hist.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig10)


        activity_stats = activity_mean_merged
        return statistics_output, keywords, activity_stats, None


    # for BERI protocol analysis:
    if beri_exists == True:
        statistics_output = percent_diff_means_no_outliers_beri, \
                            percent_diff_stddev_no_outliers_beri, percent_diff_stderr_no_outliers_beri, percent_diff_medians_no_outliers_beri, \
                            total_time, total_time_seconds
        percent_diff_means_idx = list(percent_diff_means_no_outliers_beri.index)
        y_pos = {key: percent_diff_means_idx[key-1] for key in range(1, (len(percent_diff_means_idx)+1), 1)}
        keywords = y_pos.values()

        fig11, ax = pl.subplots( nrows=1, ncols=1 )
        pl.scatter(range(0,len(beri_data['class_subject_code'])), beri_data['total_eng'], c = 'k', marker='o', s=3, label='# students engaged')
        pl.scatter(range(0,len(beri_data['class_subject_code'])), beri_data['total_diseng'], c = 'r', marker='v', s=3, label="# students disengaged")
        pl.yticks(fontsize=8)
        pl.legend(loc='upper left')
        pl.ylabel('# students')
        pl.xlabel('Observation')
        pl.ylim(0,20)
        pl.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        #pl.xlim(0, len(beri_data.index))
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.2)
        pl.tight_layout()
        fig11.savefig(os.path.join(output_dir, 'number_engaged_students.pdf'), dpi = pref_dpi)
        pl.close(fig11)

        fig12, ax = pl.subplots( nrows=1, ncols=1 )
        pl.bar(list(y_pos.keys()), percent_diff_means_no_outliers_beri, yerr=percent_diff_stderr_no_outliers_beri, error_kw=dict(lw=0.65, capsize=2, capthick=0.55), align='center', color=[0.17,0.74,0.89], alpha=1)
        pl.xticks(list(y_pos.keys()), list(y_pos.values()), rotation=90, fontsize=6)
        pl.ylim(min((percent_diff_means_no_outliers_beri-percent_diff_stderr_no_outliers_beri-10)), max(percent_diff_means_no_outliers_beri+percent_diff_stderr_no_outliers_beri+15))
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.22, left=0.12)
        pl.tight_layout()
        pl.ylabel('Mean skin conductance % difference w/o outliers\n(activity - baseline), only engaged behaviors', fontsize=6)
        pl.yticks(fontsize=6)
        if separate_baseline == True :
            fig12.savefig(os.path.join(output_dir, 'activity_means_no_outliers_separate_BL_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True :
            fig12.savefig(os.path.join(output_dir, 'activity_means_no_outliers_continuous_BL_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig12.savefig(os.path.join(output_dir, 'activity_means_no_outliers_entire_semester_BL_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig12)

        # histogram
        fig13, ax = pl.subplots( nrows=1, ncols=1 )
        pl.hist(percent_diff_means_no_outliers_beri[np.isfinite(percent_diff_means_no_outliers_beri)].values, bins=26, color=[0.85,0.33,0], align='mid', rwidth=0.92)
        pl.ylabel('Counts')
        pl.xlabel("Mean skin conductance % difference from baseline, no outliers")
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.22, left=0.12)
        pl.tight_layout()
        if separate_baseline == True :
            fig13.savefig(os.path.join(output_dir, 'activity_means_no_outliers_separate_BL_hist_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True :
            fig13.savefig(os.path.join(output_dir, 'activity_means_no_outliers_continuous_BL_hist_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig13.savefig(os.path.join(output_dir, 'activity_means_no_outliers_entire_semester_BL_hist_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig13)


        fig14, ax = pl.subplots( nrows=1, ncols=1 )
        pl.bar(list(y_pos.keys()), percent_diff_medians_no_outliers_beri, align='center', color=[0.12,0.35,1], alpha=1)
        pl.xticks(list(y_pos.keys()), list(y_pos.values()), rotation=90, fontsize=6)
        pl.ylim(min((percent_diff_medians_no_outliers_beri-10)), max(percent_diff_medians_no_outliers_beri+15))
        pl.margins(0.01,0)
        pl.subplots_adjust(bottom=0.25, left=0.15)
        pl.tight_layout()
        pl.yticks(fontsize=6)
        pl.ylabel('Median skin conductance % difference w/o outliers\n(activity - baseline), only engaged behaviors', fontsize=6)
        pl.yticks(fontsize=6)
        if separate_baseline == True :
            fig14.savefig(os.path.join(output_dir, 'activity_medians_no_outliers_separate_BL_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        elif continuous_baseline == True :
            fig14.savefig(os.path.join(output_dir, 'activity_medians_no_outliers_continuous_BL_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        else:
            fig14.savefig(os.path.join(output_dir, 'activity_medians_no_outliers_entire_semester_BL_beri.pdf'), dpi = pref_dpi, bbox_inches='tight')
        pl.close(fig14)

        activity_stats = activity_mean_merged_beri
        return statistics_output, keywords, activity_stats, beri_df




def save_output_csv(statistics_output, output_dir, keywords, activity_stats, beri_exists):
    """
    Input: Activity names ('keywords'), list of mean and median percent differences between baseline and activity
    skin conductance ('statistics_output'), and output directory where everything is saved ('output_dir')

    Goal: Save statistics as .csv file

    What it does: Creates a .csv file with one column for each statistic (e.g., mean, median, etc.)
    Each row is the statistics output for each activity (i.e., Row 1 is for the first activity)
    .csv file will be saved to output directory
    """

    filename = "skin_conductance_statistics.csv"

    if beri_exists == True:
        cols = [keywords, statistics_output[0], statistics_output[1], statistics_output[2], statistics_output[3], \
                statistics_output[4], statistics_output[5]]
        out_df = pd.DataFrame(cols)
        out_df = out_df.T
        out_df.to_csv(os.path.join(output_dir, filename), index=False, header=['Activity', 'Mean % diff, BERI no outliers', 'Std. dev. of mean % diff, BERI no outliers',\
                    'Std. err. of mean % diff, BERI no outliers', 'Median % diff, BERI no outliers', 'Total time', 'Total time (sec)'])

    elif beri_exists == False:
        cols = [keywords, statistics_output[0], statistics_output[1], statistics_output[2], statistics_output[3], \
                statistics_output[4], statistics_output[5]]
        out_df = pd.DataFrame(cols)
        out_df = out_df.T
        out_df.to_csv(os.path.join(output_dir, filename), index=False, header=['Activity', 'Mean % diff, no outliers', 'Std. dev. of mean % diff, no outliers', \
                    'Std. err. of mean % diff, no outliers', 'Median % diff, no outliers', 'Total time', 'Total time (sec)'])

    # raw skin conductance values for each sensor id, each activity
    export_csv = activity_stats.to_csv(os.path.join(output_dir, 'activity_stats.csv'), index = None, header=True)

    if beri_exists == True:
        cols_to_keep = ['class_date_time','total_eng','total_diseng']
        #export_beri = beri_df[cols_to_keep].to_csv(os.path.join(output_dir, 'beri_protocol_stats.csv'), index = None, header=True)

    print("Saved all files to output directory")
    print(" ")

    return filename



def format_and_plot_data(working_dir, timing_xcel, sheetname, beri_exists, beri_files, Fs, delta, pref_dpi, separate_baseline, continuous_baseline, grade_files, grades_exist):
    """
    Goal: Format all data downloaded from empatica website, plot data, and save statistics

    What it does: Changes to working directory and runs all above functions in one go
    Produces the four figures and statistics .csv file
    """

    try:
        timing_xcel = str(timing_xcel)
        sheetname = str(sheetname)
        Fs = int(Fs)
        delta = float(delta)
        pref_dpi = float(pref_dpi)
        separate_baseline = eval(separate_baseline)
        continuous_baseline = eval(continuous_baseline)
        beri_exists = eval(beri_exists)
    except:
        print('Fs, delta, and pref_dpi must be floating point numbers; timing_xcel, sheetname must be strings')

    # changes to working directory
    os.chdir(working_dir)
    # makes an output directory inside working directory for all saved output files (figures, csv)
    output_dir = os.path.join(os.path.split(working_dir)[0], 'output_dir')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    zip_list, EDA_list, HR_list, tag_list = extract_zip_format_filenames(working_dir)
    print('Parsed ' + str(len(zip_list)) + ' zip archives ')
    print(" ")

    EDA_dataframe_list = []
    fullRecordTime = []
    idx = 0

    for EDA_file in EDA_list:

        # 1. read EDA.csv file
        eda_df = pd.read_csv(EDA_file, header=2, names=['skin_conduct'])

        # 2. extract initial timestamp from the sensor name
        initTimestamp = int(EDA_file[-25:-15])

        # 3. throw error message if timestamp is too short/long
        if (len(str(initTimestamp)))!= 10:
            raise Exception('Error: not enough digits in timestamp')

        # 4. append all EDA data into single list, separate columns
        EDA_dataframe_list.append((eda_df, initTimestamp))

    for idx, (data, initTimestamp) in enumerate(EDA_dataframe_list):
            fullRecordTime = []
            for data_idx in range(len(data)):
                fullRecordTime.append(data_idx * 0.25)

            fullRecordTime = [datetime.fromtimestamp(x + initTimestamp) for x in fullRecordTime]

            data['timestamp'] = fullRecordTime

            EDA_dataframe_list[idx] = data

    EDA_data_df = pd.concat(EDA_dataframe_list,keys=[os.path.basename(name) for name in EDA_list])
    EDA_data_df['sensor_ids'] = EDA_data_df.index.get_level_values(0).str.split('_').str[1]

    beri_df, beri_data = get_beri_protocol(working_dir, beri_files, beri_exists)
    EDA_data_df['beri_obs'] = np.nan


    def eda_beri_merge(row, beri_df):
        out_val = np.NaN
        if row.name in beri_df.index:
            beri_row = beri_df.loc[row.name]

            if row['sensor_ids'] in beri_row:
                out_val = beri_row[row['sensor_ids']]

        return out_val

    if beri_exists == True:
        EDA_data_df2 = EDA_data_df.reset_index()
        EDA_data_df2 = EDA_data_df2.set_index('timestamp')
        EDA_data_df2['beri_obs'] = EDA_data_df2.apply(eda_beri_merge, axis=1, beri_df=beri_df)
        EDA_data_df2 = EDA_data_df2.reset_index()

        # mask EDA_data_df2 to only include skin conductance during "engaged" behaviors
        filter1 = EDA_data_df2["skin_conduct"] > 0
        filter2 = EDA_data_df2["beri_obs"] != False
        EDA_data_df2.where(filter1 & filter2, inplace = True)
        # EDA_data_df2_to_save = EDA_data_df2.drop(['level_0', 'level_1'], axis=1)
        # EDA_data_df2_to_save.to_csv('EDA_out_with_beri2.csv')
    else:
        EDA_data_df2 = []

    EDA_by_sensor = EDA_data_df.reset_index().drop('level_0', axis = 1).groupby('sensor_ids')['skin_conduct'].mean()

    # skin conductance for decomposition analysis
    obs_EDA = EDA_data_df.iloc[0:len(EDA_dataframe_list[0])]["skin_conduct"]

    # def format_cvx(grp, Fs):
    #     grp['phasic'], grp['tonic'], grp['residuals'] = cvxEDA(grp['skin_conduct'], 1./Fs)
    #     return grp

    #EDA_data_df = EDA_data_df.groupby(level=0).apply(format_cvx, Fs=Fs)

    # print("Completed phasic/tonic components")
    # print(" ")

    #cvx_first = EDA_data_df.groupby(level=0).first()

    #statistics_output, keywords, activity_stats, beri_df = plot_results(cvx_first['skin_conduct'], cvx_first['phasic'], cvx_first['tonic'], Fs, pref_dpi, EDA_data_df, output_dir, separate_baseline)
    statistics_output, keywords, activity_stats, beri_df = plot_results(Fs, pref_dpi, EDA_data_df, EDA_data_df2, output_dir, separate_baseline, continuous_baseline, beri_exists, EDA_by_sensor, grades_exist)

    if grades_exist == True:
        sep_grades_df, clicker_q_df, grades = get_grades(working_dir, grade_files, EDA_by_sensor)

    save_output_csv(statistics_output, output_dir, keywords, activity_stats, beri_exists)

    return beri_df


if __name__=='__main__':
    working_dir, timing_xcel, sheetname, beri_exists, beri_files, Fs, delta, pref_dpi, separate_baseline, continuous_baseline, grade_files, grades_exist = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12]
    format_and_plot_data(working_dir, timing_xcel, sheetname, beri_exists, beri_files, Fs, delta, pref_dpi, separate_baseline, continuous_baseline, grade_files, grades_exist)
