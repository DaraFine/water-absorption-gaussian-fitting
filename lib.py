import pandas as pd
import numpy as np
import pickle
from itertools import compress
import datetime
import pathlib

def load_exp_files_temp_experiment(exp_files):
    darks = []
    lights = []
    comments = []
    times = []
    specs = []
    
    for file in exp_files:
        
        data, dark, light, comment, time = load_exp_file_experiment(file)
        for string in range(0, len(data)):
            spec = data[string]
            specs.append(spec)
            lights.append(light)
            comments.append(comment)
            times.append(time)
            darks.append(dark)
            #print(len(specs))       
    dict_df = {'spectra': specs, 'name': comments, 'dark': darks, 'light': lights, 'time': times}

    df_all = pd.DataFrame(dict_df)
    
    return df_all

def load_exp_file_Record(file):
    with pd.HDFStore(file, mode='r') as f:
        list_keys = list(f.keys())
#         print(list_keys)
        sp = f.get('data')
        d = f.get('dark')
        l = f.get('light')
        com = f.get('comment')
        settings = f.get('settings')
        time = f.get('timestamp')

    time = file.stem[:15]
    time = datetime.datetime.strptime(time, '%H_%M_%S_%f').time()
    comment = com

    if str(type(d)) == "<class 'pandas.core.frame.DataFrame'>":
        dark = d.mean()
    else:
        dark = d
    if str(type(l)) == "<class 'pandas.core.frame.DataFrame'>":
        light = l.mean()
    else:
        light = l
    
    data = sp
    
    return data, dark, light, comment, settings, time

def load_exp_files_temp_Record(exp_files):
    darks = []
    comments = []
    times = []
    specs = []
    lights = []
    sets = []
    
    for file in exp_files:
        
        data, dark, light, comment, settings, time = load_exp_file_Record(file)
        if data.shape ==(1,2024) or comment.shape==(1,):
            #print(data.shape, dark.shape, light.shape, comment.shape, settings.shape, time)
            spec = data.T.iloc[:,0]
            specs.append(spec)
            comments.append(comment[0])
            times.append(time)
            darks.append(dark)
            lights.append(light)
            sets.append(settings)
            
        else:
            #print(data.shape, dark.shape, light.shape, comment.shape, settings.shape, time)
            for str_number in range(0, len(data)):
                spec = data.iloc[str_number]
                specs.append(spec)
                comments.append(comment[0])
                times.append(time)
                darks.append(dark)
                lights.append(light)
                sets.append(settings)
           
    dict_df = {'spectra': specs, 'name': comments, 'dark': darks, 'light': lights, 'time': times, 'settings': sets}
    df_all = pd.DataFrame(dict_df)
    
    return df_all

def load_exp_file_experiment(file):
    with pd.HDFStore(file, mode='r') as f:
        list_keys = list(f.keys())
#         print(list_keys)
        spec = f.get('exp_drs')
        d = f.get('dark')
        met = f.get('metadata')
        wvl = f.get('wavelengths')
        l = f.get('light')
        
    time = file.stem[:15]
    time = datetime.datetime.strptime(time, '%H_%M_%S_%f').time()
    
    comment = met.iloc[0,0]
    if str(type(d)) == "<class 'pd.core.series.Series'>" or "<class 'pandas.core.frame.DataFrame'>":
        dark = d.mean()
    else:
        dark = d
    if str(type(l)) == "<class 'pd.core.series.Series'>" or "<class 'pandas.core.frame.DataFrame'>":
        light = l.mean()
    else:
        light = l 
    data = []
    for string in range(0, len(spec)-4):
        sub_data = pd.Series(spec.values[string], index=wvl.values)
        data.append(sub_data)
    #data = list(data)
    
    return data, dark, light, comment, time

def load_exp_files_temp_general(exp_files):
    record_files, experiment_files = [], []
    for file in exp_files:
        
        with pd.HDFStore(file, mode='r') as f:
            list_keys = list(f.keys())
            if list_keys == ['/comment', '/dark', '/data', '/light', '/settings', '/timestamp']:
                record_files.append(file)
            elif list_keys == ['/dark', '/exp_drs', '/light', '/metadata', '/wavelengths']:
                experiment_files.append(file)
    #print(len(record_files), len(experiment_files))
    data_1 = load_exp_files_temp_Record(record_files)
    data_2 = load_exp_files_temp_experiment(experiment_files)
    return data_1, data_2