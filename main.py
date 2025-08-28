#import pandas_patches
#pandas_patches.patch()
#%pylab inline
import pandas as pd
import pathlib
import numpy as np
import sys
import os
#import lab_tools.plot_utils as pu
#pu.load_figure_style()
#import occlusion as oc
import importlib
#import surgical_fluorimeter as sf
#import surgical_processor as sp
#import sync
import matplotlib.pyplot as plt
import datetime
#import temp_lib as tmpr
#import temp_drs as tdrs
#from datetime import timedelta
from lmfit.models import PolynomialModel
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error as mse
#import lib_thyroid.data as td
#import lib_thyroid.metadata as tmd
#import lib_thyroid.spectra_processor as tsp
import lib
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['figure.figsize'] = (6.4, 4.8)
mpl.rcParams['axes.titlesize'] = 16

def get_temp(name):
    if 'bg' in name:
        return 28
    a = name.split('_')[1]
    if len(a) == 2:
        return int(a)
    return int(a[:2])

def get_time(name):
    if 'after' in name:
        return 'after'
    return 'before'

def get_name(name):
    return name[:-5]
    
def norm(spec):
    return (spec - spec.min()) / (spec.max() - spec.min())

def smooth(spec):
    return spec.rolling(window=10, center=True).mean()

def wvl_to_k(wvl, wvl_exc=660):
    k = (1/wvl_exc - 1/wvl) * 1e7
    return k

def scale_range(spec1, t_max, t_min):
    return (spec1 - spec1.min()) / (spec1.max() - spec1.min()) * (t_max - t_min) + t_min

def time_to_timedelta(t):
    t_delta = datetime.timedelta(hours = t.hour, minutes = t.minute, seconds = t.second, microseconds = t.microsecond)
    return t_delta

def filter_spectra(s):
    if (s.loc[819:820].mean() - s.loc[879:880].mean() > 0) or (s.loc[879:880].mean() < 20000):
        return True
    else:
        return False
    
def approx_cub_parab(s):
    model = PolynomialModel(degree = 3)
    pars = model.make_params(c0 = 1, c1 = 1, c2 = 1, c3 = 1)
    fit_wvls = np.hstack([s.loc[:820].index.values, s.loc[880:890].index.values])
    fit_intens = np.hstack([s.loc[:820].values, s.loc[880:890].values])
    result = model.fit(fit_intens, pars, x = fit_wvls)
    xs = s.loc[:890].index.values
    approx = pd.Series(result.eval(x = xs), xs)
    return approx

def peak_ratio(s):
    s_without_bg = s.loc[:890] - approx_cub_parab(s)
    peak_1 = s_without_bg.loc[line_1 - delta: line_1 + delta].mean()
    peak_2 = s_without_bg.loc[line_2 - delta: line_2 + delta].mean()
    return peak_2/peak_1

def approx_line(s):
    model = PolynomialModel(degree = 1)
    pars = model.make_params(c0 = 1, c1 = 1)
    fit_temp = s.index.values
    fit_peak_ratio = s.values
    result = model.fit(fit_peak_ratio, pars, x = fit_temp)
    approx = pd.Series(result.eval(x = fit_temp), fit_temp)
    return approx

def result_approx_line(s):
    model = PolynomialModel(degree = 1)
    pars = model.make_params(c0 = 1, c1 = 1)
    fit_temp = s.index.values
    fit_peak_ratio = s.values
    result = model.fit(fit_peak_ratio, pars, x = fit_temp)
    return result

def normed_graph(s):
    x_min = s.min()
    x_max = s.max()
    normed_s = s.apply(lambda x: (x - x_min)/(x_max - x_min))
    return normed_s

def norm(spec):
    return (spec - spec.min()) / (spec.max() - spec.min())

def smooth(spec):
    return spec.rolling(window=30, center=True).mean()

def filters_files(files, garbage_names):
    stems = [name_to_stem(file) for file in files]
    mask = np.array([False if ele in garbage_names else True for ele in stems])
    masked_files = list(compress(files, mask))
    return masked_files

def preprocess(sp):
    sp = sp.iloc[5:-5]
    sp = sp.rolling(window=5, center=True).median()
    sp = sp - sp.loc[250:275].mean()
    return sp.dropna()

def calc_OD(ser):
    ser = ser.iloc[:len(light_ser)]
    num = (ser - preprocessed_dark)
    denum = preprocessed_light - preprocessed_dark
    denum[denum < 1e-9] = 1e-9
    r = num / denum
    mask = r < 1e-9
    r[mask] = 1e-9
    return -np.log10(r) 
def load_spectra_data(path, mask='*.hdf'):
    return sorted(path.glob(mask))

def divide_on_2(name):
    parts = name.split('_')
    name = parts[0]+'_'+'1.5mm'+'_'+parts[2]
    
    return name

def preprocess(sp, keep_series=False):
    sp = sp.iloc[5:-5]
    sp = sp.rolling(window=5, center=True).median()
#     sp = sp - sp.loc[250:275].mean()
    return sp.dropna()

def calc_OD(ser):
    ser = ser.iloc[:len(light)]
    ser = ser.rolling(window=5, center=True).median()
    num = (ser - preprocessed_dark)
    denum = preprocessed_light - preprocessed_dark
    denum[denum < 1e-9] = 1e-9
    r = num / denum
    mask = r < 1e-9
    r[mask] = 1e-9
    result = -np.log10(r)
    return result

def ratio(ser):
    ser = ser.rolling(window=50, center=True).mean()
    ser = ser - straight_line(ser,930,1060)
    ser /= ser.loc[970:975].mean()
    ser = ser.loc[800:1060]
    rat = (ser.loc[990:1000].mean()/10)/(ser.loc[970:975].mean()/5)
    
    return rat

def difference(ser):
    ser = ser.loc[800:1000]
    ser = ser - ser.loc[800:850].median()
    ser = ser.rolling(window=50, center=True).mean()
    ser /= ser.loc[970:975].mean()
    diff = (ser.loc[990:1000].mean()/10)-(ser.loc[970:975].mean()/5)

    return diff

def temp(name):
    t = float(name.split('_')[-1])
    return float(t)  

def sum_of_gaussians(x, params):
    
    assert len(params) % 3 == 0
    result = np.zeros_like(x)
    n_curves = len(params) // 3  # Количество гауссовых кривых

    for i in range(n_curves):
        amplitude = params[i * 3]          # Амплитуда
        mean = params[i * 3 + 1]          # Среднее значение
        std_dev = params[i * 3 + 2]       # Стандартное отклонение

        result += amplitude * np.exp(-((x - mean) ** 2) / (2 * ((std_dev) ** 2)))

    return result

def find_local_maximum(series, step):
    
    local_maxima = []
    data_max_x = []
    x_data = series.index
    for i in range(step+1, len(series)-step-1, 1):
        y = series.iloc[i]
        x = x_data[i]
        count = 1
        for j in range(i-step, i+step):
            if y<series.iloc[j]:
                count = 0
        if y == series.iloc[i-1]: count = 0
        if count == 1:
            local_maxima.append(y)
            data_max_x.append(x)
            
    return local_maxima, data_max_x


data_folder_spec = 'data_1'
path = pathlib.Path(data_folder_spec)
files = load_spectra_data(path, mask='*.hdf')
data_1, data_2 = lib.load_exp_files_temp_general(files)
data = pd.concat([data_1.drop(columns = 'settings'), data_2], axis=0)
data.reset_index(drop=True, inplace=True)

dark = data.dark.iloc[0]
light = data.light.iloc[0]

df_new = data
df_new.groupby('name')


preprocessed_light = preprocess(light)
preprocessed_dark = preprocess(dark)
wls = (df_new['spectra'].apply(preprocess, keep_series=True)).iloc[0].index.values

preprocessed_light.index = wls[:len(preprocessed_light)]
preprocessed_dark.index = wls[:len(preprocessed_dark)]

preprocessed_list = []
for spectrum in df_new['spectra']:
    preprocessed_list.append(preprocess(spectrum))      
df_new['preprocessed'] = preprocessed_list

od_list = []
for per in df_new['preprocessed']:
    od_value = calc_OD(per)
    od_list.append(od_value)
df_new['OD_preprocessed'] = od_list

a = 0
dff = pd.DataFrame()

for i in range(0, len(df_new)):
    sp = df_new.spectra.iloc[i]
    s = calc_OD(df_new.spectra.iloc[i])
    if (filter_spectra(sp)==True) or (s.loc[1000:1050].mean() > 2):
        a += 1
    else:
        dff = pd.concat([dff, df_new.iloc[i].to_frame().T], ignore_index=True)
    
df_new = pd.DataFrame()
df_new = dff
print(df_new.columns)

OD = df_new.OD_preprocessed[1]
OD1 = (smooth(OD).dropna())
for i in OD1.index:
            if i >= 850:
                ind = i
                break
for i in OD1:
    OD1.replace(i, i-OD[ind], inplace=True)

peak_value1, peak_index1 = find_local_maximum(OD1,170)
