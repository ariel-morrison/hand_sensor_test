# Script to unzip downloaded EDA files from Empatica website, analyze skin conductance data, plot data, and save output as csv

# the following packages are for opening/reading files and analyzing data
import os
import sys
import zipfile
import shutil
import pandas as pd
import numpy as np
import cvxopt as cv
import cvxopt.solvers
import statistics
from statistics import mean
import datetime
from datetime import datetime
import time
import csv

# the following packages are for plotting
import pylab as pl
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


def cvxEDA(obs_EDA, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,
           solver=None, options={'reltol': 1e-9, 'show_progress': False}):

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
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
                'reltol' = relative accuracy
                'abstol' = absolute accuracy
                'feastol' = tolerance for feasibility conditions

    Returns (see paper for details):
       phasic = phasic component
       p: sparse SMNA driver of phasic component
       tonic: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)

       from Greco et al. (2016). cvxEDA: A Convex Optimization Approach
        to Electrodermal Activity Processing, IEEE Transactions on Biomedical
        Engineering, 63(4): 797-804.
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

    # Solve the problem:
    # .5*(M*q + B*l + C*d - obs_EDA)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

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

    return (np.array(a).ravel() for a in (phasic, p, tonic, l, d, e, obj))


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

                # if the unzipped folder already exists, skip the unzipping process
                # if os.path.exists(zipfile_name):
                #     print('Zip archive ' + zipfile_name + ' is unzipped.')

                sensorNum = path_to_zip_file[-21:-4]

                # Check current working directory.
                working_sub_dir = os.path.join(working_dir, sensorNum)

                eda_filepath = os.path.join(working_sub_dir, 'EDA.csv')
                if os.path.isfile(eda_filepath): # check if an EDA.csv file exists in the folder
                    eda_filename = working_dir + '/' + str(sensorNum) + '_EDA.csv'
                    os.rename(eda_filepath, eda_filename)
                    EDA_list.append(eda_filename)

                if os.path.isfile(os.path.join(working_sub_dir, 'HR.csv')): # check if a HR.csv file exists in the folder
                    hr_filename = working_dir + '/' + str(sensorNum) + '_HR.csv'
                    os.rename(working_sub_dir + '/' + 'HR.csv', hr_filename)
                    HR_list.append(hr_filename)

                if os.path.isfile(os.path.join(working_sub_dir, 'tags.csv')): # check if a tags.csv file exists in the folder
                    tag_filename = working_dir + '/' + str(sensorNum) + '_tags.csv'
                    os.rename(working_sub_dir + '/' + 'tags.csv', tag_filename)
                    tag_list.append(tag_filename)

                shutil.rmtree(working_sub_dir)

    return zip_list, EDA_list, HR_list, tag_list



def get_activity_timing(working_dir, timing_xcel, sheetname, EDA_data_df):
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
    within the start and end time of each component.
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

    x_out = xcel.apply(lambda row : EDA_data_df[(EDA_data_df['timestamp']>=row['datetime_start'])&(EDA_data_df['timestamp']<row['datetime_end'])].assign(activity=row['Activity Start']), axis=1)

    activity_diff = pd.concat(list(x_out)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].max() - 2*(pd.concat(list(x_out)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].min())
    bad_baseline = activity_diff.reset_index()
    #print(bad_baseline)
    activity_mean = pd.concat(list(x_out)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].mean()
    activity_stddev = pd.concat(list(x_out)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].std()
    activity_stderr = pd.concat(list(x_out)).reset_index().groupby(['level_0', 'activity'])['skin_conduct'].sem()

    return activity_mean, activity_stddev, activity_stderr




def get_beri_protocol(working_dir, timing_beri):
    """
    Input: working directory (working_dir) where all data are downloaded from Empatica website;
            spreadsheet (timing_beri) where BERI protocol observations are recorded (see example)

    Goal: Find how many students exhibited engaged/disengaged behaviors

    What it does: Opens the spreadsheet where all BERI observations are recorded, sums the number of
    engaged/disengaged students during each type of activity, then normalizes it by the number of
    instances of that activity
    """

    os.chdir(working_dir)

    beri_timing = os.path.join(working_dir, "beri_example.xlsx")
    beri = pd.read_excel(beri_timing)

    beri_group = beri.groupby(["Instructor activity"])
    eng_students_agg = beri_group[['# students engaged', '# students disengaged']]
    eng_students_agg = eng_students_agg.aggregate(
        {'# students engaged': ['mean', 'sem', 'std'],
         '# students disengaged': ['mean', 'sem', 'std']
        })
    eng_students_agg.reset_index(level = 0, inplace = True)
    eng_students_agg.columns.set_levels(["mean","std. error","std. dev",""],level=1,inplace=True)

    return eng_students_agg



def plot_results(y, r, p, t, l, d, e, obj, Fs, pref_format, pref_dpi, EDA_data_df, output_dir):
    """
    Input: for plotting an individual's data - skin conductance dataframe (obs_EDA), phasic/tonic components,
    coefficients of tonic spline (l), offset and slope of the linear drift term (d), model residuals (e), value
    of objective function being minimized (obj)
    Sampling frequency per second (Fs), preferred figure format (pref_format), preferred figure resolution (pref_dpi)

    Goal: To produce figures and save them to output directory

    What it does: Plots line graphs of an individual's total, phasic, and tonic components of skin conductance
    against minutes. Calculates percent difference in mean skin conductance between an activity and baseline, plots
    bar graph for percent difference for each activity.
    """

    timing = pl.arange(1., len(y) + 1.) / (60 * Fs) # minutes = divide by 240 = 60 seconds * 4 records/sec

# plotting total conductance (phasic + tonic + noise)
    fig1, ax = pl.subplots( nrows=1, ncols=1 )
    pl.plot(timing, y, color = 'r')
    pl.xlim(0, max(timing) + 1)
    pl.ylabel('Skin conductance - total (\u03bcS)')
    pl.xlabel('Time (min)')
    fig1.savefig(os.path.join(output_dir, 'total_conductance.png'), format = pref_format, dpi = pref_dpi)
    pl.close(fig1)

# plotting phasic component of skin conductance
    ylim_top = max(r)
    fig2, ax = pl.subplots( nrows=1, ncols=1 )
    pl.plot(timing, r, color = 'b')
    pl.xlim(0, max(timing) + 1)
    pl.ylabel('Skin conductance - phasic component (\u03bcS)')
    pl.xlabel('Time (min)')
    fig2.savefig(os.path.join(output_dir, 'phasic_component.png'), format = pref_format, dpi = pref_dpi)
    pl.close(fig2)

# plotting tonic component of skin conductance
    ylim_top = max(t)
    fig3, ax = pl.subplots( nrows=1, ncols=1 )
    pl.plot(timing, t, color = 'g')
    pl.xlim(-1, max(timing) + 1)
    pl.ylabel('Skin conductance - tonic component (\u03bcS)')
    pl.xlabel('Time (min)')
    fig3.savefig(os.path.join(output_dir, 'tonic_component.png'), format = pref_format, dpi = pref_dpi)
    pl.close(fig3)

    # get timing and EDA for each activity
    activity_mean, activity_stddev, activity_stderr = get_activity_timing(working_dir, timing_xcel, sheetname, EDA_data_df)
    activity_mean = activity_mean.reset_index()
    activity_mean = activity_mean.rename(columns={'level_0': 'sensor_id'})
    activity_stddev2 = activity_stddev.reset_index().rename(columns = {"level_0":"sensor_id","skin_conduct":"stddev_skin_conduct"})
    activity_stderr = activity_stderr.reset_index()
    activity_stderr2 = activity_stderr.rename(columns = {"level_0":"sensor_id","skin_conduct":"stderr_skin_conduct"})
    activity_stats = pd.concat([activity_mean, activity_stddev2['stddev_skin_conduct'], activity_stderr2['stderr_skin_conduct']], axis=1)

    # take out baselines for each sensor/person
    # each activity only compared against that person's baseline
    # --> still need to eliminate data records where baseline doesn't fit criterion

    baselines = activity_mean[activity_mean['activity'] == "Baseline"][["sensor_id", "skin_conduct"]]
    baselines = baselines.rename(columns = {"skin_conduct":"skin_conduct_baseline"})
    activity_mean_no_bl = activity_mean[activity_mean['activity'] != "Baseline"]
    activity_mean_no_bl = activity_mean_no_bl.rename(columns = {"skin_conduct":"skin_conduct_means"})
    activity_mean_merged = activity_mean_no_bl.merge(baselines, on = ["sensor_id"])


    # mean/median percent difference between baseline and activity
    percent_diff_means = activity_mean_merged.groupby(['activity']).apply(lambda row: ((row["skin_conduct_means"] - row["skin_conduct_baseline"])/row["skin_conduct_baseline"]).mean()*100)
    percent_diff_medians = activity_mean_merged.groupby(['activity']).apply(lambda row: ((row["skin_conduct_means"] - row["skin_conduct_baseline"])/row["skin_conduct_baseline"]).median()*100)
    percent_diff_stddev = activity_mean_merged.groupby(['activity']).apply(lambda row: ((row["skin_conduct_means"] - row["skin_conduct_baseline"])/row["skin_conduct_baseline"]).std()*100)
    percent_diff_stderr = activity_mean_merged.groupby(['activity']).apply(lambda row: ((row["skin_conduct_means"] - row["skin_conduct_baseline"])/row["skin_conduct_baseline"]).sem()*100)

    # for statistics csv output
    statistics_output = percent_diff_means, percent_diff_medians, percent_diff_stddev, percent_diff_stderr

    # for plotting on the same axes for all bar graphs
    INT_MAX = sys.maxsize
    INT_MIN = -sys.maxsize - 1
    y_bottom = min(min(percent_diff_means, default=INT_MAX), min(percent_diff_medians, default=INT_MAX))
    y_top = max(max(percent_diff_means, default=INT_MIN), max(percent_diff_medians, default=INT_MIN))

    # x-axis labels = activity names
    percent_diff_means_idx = list(percent_diff_means.index)
    y_pos = {key: percent_diff_means_idx[key-1] for key in range(1, (len(percent_diff_means_idx)+1), 1)}
    keywords = y_pos.values()

    # mean percent difference
    fig4, ax = pl.subplots( nrows=1, ncols=1 )
    pl.bar(list(y_pos.keys()), percent_diff_means, align='center', color=[0.25,0.45,0.5], alpha=1)
    pl.xticks(list(y_pos.keys()), list(y_pos.values()), rotation=90)
    if (0-0.5) <= y_bottom <= 0.5:
        pl.ylim(y_bottom, y_top+10)
    else:
        pl.ylim(y_bottom-5, y_top+10)
    pl.margins(0.15)
    pl.subplots_adjust(bottom=0.2)
    pl.ylabel('Mean skin conductance % difference\n(activity - baseline)')
    fig4.savefig(os.path.join(output_dir, 'activity_means.png'), format = pref_format, dpi = pref_dpi)
    pl.close(fig4)

    # median percent difference
    fig5, ax = pl.subplots( nrows=1, ncols=1 )
    pl.bar(list(y_pos.keys()), percent_diff_medians, align='center', color=[0.12,0.35,1], alpha=1)
    pl.xticks(list(y_pos.keys()), list(y_pos.values()), rotation=90)
    if (0-0.5) <= y_bottom <= 0.5:
        pl.ylim(y_bottom, y_top+10)
    else:
        pl.ylim(y_bottom-5, y_top+10)
    pl.margins(0.15)
    pl.subplots_adjust(bottom=0.2)
    pl.ylabel('Median skin conductance % difference\n(activity - baseline)')
    fig5.savefig(os.path.join(output_dir, 'activity_medians.png'), format = pref_format, dpi = pref_dpi)
    pl.close(fig5)



    # for BERI protocol analysis:
    eng_students_agg = get_beri_protocol(working_dir, timing_beri)

    eng_students_agg_idx = list(eng_students_agg['Instructor activity'])
    eng_students_agg_idx = {key: eng_students_agg_idx[key] for key in range(0, (len(eng_students_agg_idx)), 1)}

    fig6, ax = pl.subplots( nrows=1, ncols=1 )
    eng_students_agg["# students engaged"]["mean"].plot.bar(x = list(eng_students_agg_idx.values()), yerr = eng_students_agg["# students engaged"]["std. dev"], color=[0.4,0.2,0.5], rot=65, capsize=5)
    pl.xticks(list(eng_students_agg_idx.keys()), list(eng_students_agg_idx.values()), rotation=45)
    pl.ylabel("# engaged students")
    pl.margins(0.15)
    pl.subplots_adjust(bottom=0.2)
    pl.tight_layout()
    fig6.savefig(os.path.join(output_dir, 'number_engaged_students.png'), format = pref_format, dpi = pref_dpi)
    pl.close(fig6)

    fig7, ax = pl.subplots( nrows=1, ncols=1 )
    eng_students_agg["# students disengaged"]["mean"].plot.bar(x = list(eng_students_agg_idx.values()), yerr = eng_students_agg["# students disengaged"]["std. dev"], color=[0,0.52,0], rot=65, capsize=5)
    pl.xticks(list(eng_students_agg_idx.keys()), list(eng_students_agg_idx.values()), rotation=45)
    pl.ylabel("# disengaged students")
    pl.margins(0.15)
    pl.subplots_adjust(bottom=0.2)
    pl.tight_layout()
    fig7.savefig(os.path.join(output_dir, 'number_disengaged_students.png'), format = pref_format, dpi = pref_dpi)
    pl.close(fig7)

    return statistics_output, keywords, activity_stats, eng_students_agg



def save_output_csv(statistics_output, output_dir, keywords, activity_stats, eng_students_agg):
    """
    Input: Activity names ('keywords'), list of mean and median percent differences between baseline and activity
    skin conductance ('statistics_output'), and output directory where everything is saved ('output_dir')

    Goal: Save statistics as .csv file

    What it does: Creates a .csv file with one column for each statistic (e.g., mean, median, etc.)
    Each row is the statistics output for each activity (i.e., Row 1 is for the first activity)
    .csv file will be saved to output directory
    """

    filename = "skin_conductance_statistics.csv"

    cols = [keywords, statistics_output[0], statistics_output[1], statistics_output[2], statistics_output[3]]
    out_df = pd.DataFrame(cols)
    out_df = out_df.T
    out_df.to_csv(os.path.join(output_dir, filename), index=False, header=['Activity', 'Mean % diff', 'Median % diff', 'Std. dev. of mean % diff', 'Std. err. of mean % diff'])

    # raw skin conductance values for each sensor id, each activity
    export_csv = activity_stats.to_csv(os.path.join(output_dir, 'activity_stats.csv'), index = None, header=True)
    export_beri = eng_students_agg.to_csv(os.path.join(output_dir, 'beri_protocol_stats.csv'), index = None, header=True)

    print("Saved all files to output directory")
    print(" ")

    return filename


def format_and_plot_data(working_dir, timing_xcel, sheetname, timing_beri, Fs, delta, pref_format, pref_dpi):
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
        pref_format = str(pref_format)
        pref_dpi = float(pref_dpi)
    except:
        print('Fs, delta, and pref_dpi must be floating point numbers, timing_xcel, sheetname, pref_format must be strings')

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
        # 2. append all EDA data into single list, separate columns
        EDA_dataframe_list.append(eda_df)

        # 2. extract initial timestamp from the sensor name
        initTimestamp = int(EDA_file[-25:-15])
        dt_object = datetime.fromtimestamp(initTimestamp)
        #print("dt_object =", dt_object)

        # 3. check that the timestamp is the right length
        checkTimestampLength = len(str(initTimestamp))

        # 4. throw error message if timestamp is too short/long
        if checkTimestampLength != 10:
            raise Exception('Error: not enough digits in timestamp')


    for idx, data in enumerate(EDA_dataframe_list):
            fullRecordTime = []
            for data_idx in range(len(data)):
                fullRecordTime.append(data_idx * 0.25)

            fullRecordTime = [datetime.fromtimestamp(x + initTimestamp) for x in fullRecordTime]

            data['timestamp'] = fullRecordTime

            EDA_dataframe_list[idx] = data

    EDA_data_df = pd.concat(EDA_dataframe_list,keys=[os.path.basename(name) for name in EDA_list])
    EDA_data_df.keys()

    # save conductance to csv file
    #export_csv = EDA_data_df.to_csv ((working_dir + '/' + 'raw_skin_conductance.csv'), index = None, header=True)

    #extract timesteps column
    obs_EDA = EDA_data_df.iloc[0:len(EDA_dataframe_list[0])]["skin_conduct"]
    obs_EDA_list = list(obs_EDA)

    phasic, p, tonic, l, d, e, obj = cvxEDA(obs_EDA_list, 1./Fs)

    statistics_output, keywords, activity_stats, eng_students_agg = plot_results(obs_EDA, phasic, p, tonic, l, d, e, obj, Fs, pref_format, pref_dpi, EDA_data_df, output_dir)
    save_output_csv(statistics_output, output_dir, keywords, activity_stats, eng_students_agg)
    return eng_students_agg



if __name__=='__main__':
    working_dir, timing_xcel, sheetname, timing_beri, Fs, delta, pref_format, pref_dpi = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]
    format_and_plot_data(working_dir, timing_xcel, sheetname, timing_beri, Fs, delta, pref_format, pref_dpi)
