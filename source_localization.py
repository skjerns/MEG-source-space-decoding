# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:44:08 2023

@author: Simon
"""
import os
import settings
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample, fetch_fsaverage
from mne.beamformer import make_lcmv, apply_lcmv_epochs
import mne
import meg_tools
import utils
import numpy as np
from load_funcs import stratify_data, load_testing_img2
import ospath
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
import seaborn as sns
from sklearn.decomposition import PCA
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# filter to apply to raw data before any operation
filter_func = f'lambda x: x.filter(0.5, None, verbose=False, n_jobs=-1)'
tmin=-0.1
tmax=0.5
clf = utils.LogisticRegressionOvaNegX(C=6, penalty='l1', neg_x_ratio=2)
clf = RandomForestClassifier(200)
# clf = LinearDiscriminantAnalysis()

folders = ospath.list_folders(settings.data_dir + 'seq12', pattern='DSMR*')

fig, axs, b0 = utils.make_fig(len(folders), bottom_plots=[1, 0, 0])
fig, axs2, b1, b2, b3 = utils.make_fig(len(folders), bottom_plots=[1, 1, 1])

df = pd.DataFrame()
df_test = pd.DataFrame()
epochs_all = []
heatmaps = {'sensor':[], 'source':[]}

for i, folder in enumerate(tqdm(folders, desc='subjects')):
    subj = f'DSMR{utils.get_id(folder)}'

    # Read the raw data
    localizer_file = f'{folder}MEG/localizer1_trans[localizer1]_tsss_mc.fif'
    fwd_filename = f'{folder}/{subj}-fwd.fif'
    if not os.path.exists(fwd_filename):
        continue

    epochs = meg_tools.load_epochs(localizer_file, event_ids=np.arange(1, 11),
                                   return_mne_obj=True, filter_func=filter_func)
    epochs_all.append(epochs)
    data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.5, method="empirical", rank='info')
    noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method="empirical", rank='info')
    fwd = mne.read_forward_solution(fwd_filename)

    filters = make_lcmv(epochs.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                        pick_ori="max-power", weight_norm="unit-noise-gain",
                        rank='info')

    # first apply on original data
    data_y = epochs.events[:,2]
    data_x_sensor = epochs.get_data()
    data_x_sensor, data_y = stratify_data(data_x_sensor, data_y)
    data_x_sensor = settings.default_normalize(data_x_sensor)
    pca_sensor = UnsupervisedSpatialFilter(PCA(100), average=False)
    data_x_sensor = pca_sensor.fit_transform(data_x_sensor)
    df_sensor = utils.get_best_timepoint(data_x_sensor, data_y, n_jobs=-2, verbose=True,
                           ex_per_fold=2, clf=clf, ova=False, subj=subj, ms_per_point=10)
    df_sensor['space'] = 'sensor'

    # now apply to source reconstructed data
    stcs = apply_lcmv_epochs(epochs, filters)
    data_y = epochs.events[:,2]
    data_x_source = np.array([stc.data for stc in stcs])
    data_x_source, data_y = stratify_data(data_x_source, data_y)
    data_x_source = zscore(data_x_source, None)
    pca_source = UnsupervisedSpatialFilter(PCA(100), average=False)
    data_x_source = pca_source.fit_transform(data_x_source)
    df_source = utils.get_best_timepoint(data_x_source, data_y, n_jobs=-2, verbose=True,
                           ex_per_fold=2, clf=clf, ova=False, subj=subj, ms_per_point=10)
    df_source['space'] = 'source'

    df_subj = pd.concat([df_sensor, df_source])
    df = pd.concat([df, df_subj])

    sns.lineplot(data=df_subj, x='timepoint', y='accuracy', hue='space', ax=axs[i])
    axs[i].set_ylim([0, 0.5])
    plt.pause(0.1)

    # transfer to learning
    learning_file = f'{folder}MEG/test_trans[localizer1]_tsss_mc.fif'
    test_x, test_y = load_testing_img2(folder)
    epochs_test = mne.EpochsArray(test_x, info=epochs.info)

    data_cov = mne.compute_covariance(epochs_test, tmin=0.01, tmax=0.5, method="empirical", rank='info')
    noise_cov = mne.compute_covariance(epochs_test, tmin=tmin, tmax=0, method="empirical", rank='info')
    fwd = mne.read_forward_solution(fwd_filename)
    stcs_test = apply_lcmv_epochs(epochs_test, filters)
    test_x_sensor = epochs_test.get_data()
    test_x_source = np.array([stc.data for stc in stcs_test])

    test_x_sensor = pca_sensor.transform(test_x_sensor)
    test_x_source = pca_source.transform(test_x_source)

    df_test_subj = pd.DataFrame()
    for train_x, test_x, name in zip([data_x_sensor, data_x_source],
                                     [test_x_sensor, test_x_source],
                                     ['sensor', 'source']):

        heatmap = utils.get_transfer_heatmap(clf,
                                             train_x, data_y,
                                             test_x, test_y,
                                             verbose=True, n_jobs=-1)
        heatmaps[name] += [heatmap]
        times = np.arange(0, test_x.shape[-1]*10, 10)
        df_test_subj = pd.concat([df_test_subj, pd.DataFrame({'accuracy':heatmap.mean(0),
                                                              'space': name,
                                                              'timepoint': times,
                                                              'subj': subj})])
    df_test = pd.concat([df_test, df_test_subj])
    sns.lineplot(data=df_test_subj, x='timepoint', y='accuracy', hue='space', ax=axs2[i])


sns.lineplot(data=df, x='timepoint', y='accuracy', hue='space', ax=b0)

sns.lineplot(data=df_test, x='timepoint', y='accuracy', hue='space', ax=b1)
sns.heatmap(np.mean(heatmaps['sensor'], 0), vmin=0.1, vmax=0.6, ax=b2)
sns.heatmap(np.mean(heatmaps['sensor'], 0), vmin=0.1, vmax=0.6, ax=b3)

stop

# ploting
FS_HOME = os.environ.get('FREESURFER_HOME')
subj_dir = settings.data_dir + 'freesurfer'
os.makedirs(subj_dir, exist_ok=True)
os.environ['SUBJECTS_DIR'] = subj_dir

evoked = epochs_all[0].average()
data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.5, method="empirical", rank='info')
noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=0, method="empirical", rank='info')
fwd_filename = f'{folders[0]}/{subj}-fwd.fif'

fwd = mne.read_forward_solution(fwd_filename)

filters_vec = make_lcmv(
    evoked.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori="vector",
    weight_norm="unit-noise-gain-invariant",
    rank=None,
)
stc = mne.beamformer.apply_lcmv(evoked, filters_vec)

lims = [0.3, 0.45, 0.6]
brain = stc.plot(
    clim=dict(kind="value", lims=lims),
    hemi="both",
    size=(600, 600),
    views=["sagittal"],
    # Could do this for a 3-panel figure:
    # view_layout='horizontal', views=['coronal', 'sagittal', 'axial'],
    brain_kwargs=dict(silhouette=True),
    # **kwargs
)