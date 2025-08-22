#import pandas_patches
#pandas_patches.patch()
#%pylab inline
import pandas as pd
import pathlib
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

def preprocess(sp):
    sp = sp.iloc[5:-5]
    sp = sp.rolling(window=5, center=True).median()
#     sp = sp - sp.loc[250:275].mean()
    return sp.dropna()

preprocessed_light = preprocess(light)
preprocessed_dark = preprocess(dark)