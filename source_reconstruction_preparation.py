#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:14:07 2022

@author: simon.kern
"""
import os
import sys
import mne
import subprocess
import settings
import ospath
import time
from tqdm import tqdm
import utils
from joblib import Parallel, delayed
from tqdm import tqdm
from mne.coreg import Coregistration
from mne.io import read_info
from subprocess import PIPE


def recon_all(folder):

    SUBJ = f'DSMR{utils.get_id(folder)}'

    subj_dir = settings.data_dir + 'freesurfer'

    if os.path.exists(f'{subj_dir}/{SUBJ}/surf/lh.white') and \
        os.path.exists(f'{subj_dir}/{SUBJ}/surf/rh.white'):
        print(f'already reconstructed for {SUBJ}')
        return

    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    MRI_FOLDER = (folder + '/MRI/').replace('//', '/')
    assert os.path.isdir(MRI_FOLDER)

    NIFTI_FILE = f'{folder}/MRI/T1.nii.gz'

    # first convert the DICOM to NIFGI
    if not os.path.exists(NIFTI_FILE):
        convert_cmd = f'dcm2nii -d n {MRI_FOLDER}'
        process = subprocess.Popen(convert_cmd, stdout=PIPE, stderr=PIPE, shell=True)
        while process.stdout.readable():
            line = process.stdout.readline()
            if not line: break
            print(line.decode().strip())
            time.sleep(0.05)
        t1_files = ospath.list_files(MRI_FOLDER, patterns='t1*gz')
        if len(t1_files)==0: return
        os.rename(t1_files[0], NIFTI_FILE)
    try:
        # now extract the regions
        recon_cmd = f'recon-all -i {NIFTI_FILE} -s {SUBJ}  -all'
        process = subprocess.Popen(recon_cmd, stdout=PIPE, stderr=PIPE, shell=True)
        lines = [f'{SUBJ}: ']
        while process.stdout.readable():
            line = process.stdout.readline().decode()
            if not line: break
            lines.append(line)
            print(line.strip())
    except Exception as e:
        print('#'*40, folder)
        import traceback
        print(traceback.format_exc())
        return f'{SUBJ} : Error {traceback.format_exc()}'

    return '\n'.join(lines)

def compute_source_space(folder, spacing='oct6', subjects_dir=None, n_jobs=1):
    subj = f'DSMR{utils.get_id(folder)}'
    src_filename = f'{folder}/{subj}-src.fif'
    if os.path.isfile(src_filename):
        print(f'already computed source space for {subj}')
        return 'done'
    try:
        src = mne.setup_source_space(subject=subj, spacing=spacing,
                                     subjects_dir=subjects_dir, n_jobs=n_jobs)
        src.save(src_filename)
    except:
        print(f'ERROR computing source space for {subj}')
        import traceback
        traceback.print_exc()
        return f'{subj} : Error {traceback.format_exc()}'
    return 'done'


def compute_bem_solution(folder, ico=4, subjects_dir=None):
    subj = f'DSMR{utils.get_id(folder)}'
    bem_filename = f'{folder}/{subj}-bem.fif'
    err = ''
    if os.path.isfile(bem_filename):
        print(f'already computed source space for {subj}')
        return 'done'
    try:
        try:
            mne.bem.make_watershed_bem(subj, subjects_dir=subjects_dir, overwrite=True)
            mne.bem.make_scalp_surfaces(subject=subj, subjects_dir=subjects_dir, no_decimate=True,
                                        force=True,
                                        overwrite=True,
                                    )
        except RuntimeError:
            print(f'BEM freesurfer solution already exists for {subj}')
            err += f'BEM freesurfer solution already exists for {subj}\n'

        surfs = mne.make_bem_model(subject=subj, ico=ico,
                                     subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(surfs)
        mne.write_bem_solution(bem_filename, bem)
    except:
        print(f'ERROR computing source space for {subj}')
        import traceback
        traceback.print_exc()
        err += f'{subj} : Error {traceback.format_exc()}'
        return err
    return 'done'

def perform_coregistration(folder, subjects_dir=None):
    """perform automated coregistration for given participant/folder"""
    subj = f'DSMR{utils.get_id(folder)}'
    trans_file = f'{folder}/{subj}-trans.fif'

    if os.path.exists(trans_file):
        print('coregistration already computed')
        return 'done'
    else:
        try:
            localizer_file = f'{folder}/MEG//localizer1_trans[localizer1]_tsss_mc.fif'
            assert os.path.exists(localizer_file)

            info = read_info(localizer_file, verbose='WARNING')

            coreg = Coregistration(info, subj, subjects_dir=subjects_dir, fiducials='estimated')

            coreg.fit_fiducials(verbose=True)  # first guess
            coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)  # initial fit
            coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
            coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)  # final fit
            # dists = coreg.compute_dig_mri_distances() * 1e3  # in mm

            mne.write_trans(trans_file, coreg.trans)
        except Exception:
            print(f'ERROR computing source space for {subj}')
            import traceback
            traceback.print_exc()
            return f'{subj} : Error {traceback.format_exc()}'

def compute_forward(folder):
    subj = f'DSMR{utils.get_id(folder)}'

    fwd_filename = f'{folder}/{subj}-fwd.fif'


    if os.path.exists(fwd_filename):
        return 'done'
    else:
        try:
            bem_filename = f'{folder}/{subj}-bem.fif'  # BEM solution
            src_filename = f'{folder}/{subj}-src.fif'  # source space
            trans_filename = f'{folder}/{subj}-trans.fif'  # MRI-> MEG mapping
            assert os.path.exists(bem_filename), f'BEM file not found {bem_filename}'
            assert os.path.exists(src_filename), f'BEM file not found {src_filename}'
            assert os.path.exists(trans_filename), f'BEM file not found {trans_filename}'

            bem = mne.read_bem_solution(bem_filename)
            src = mne.read_source_spaces(src_filename)

            fif_file = f'{folder}/MEG/localizer1_trans[localizer1]_tsss_mc.fif'
            raw = mne.io.read_raw(fif_file)

            fwd = mne.make_forward_solution(raw.info, trans_filename, src, bem)
            mne.write_forward_solution(fwd_filename, fwd)

        except Exception:
            print(f'ERROR computing source space for {subj}')
            import traceback
            traceback.print_exc()
            return f'{subj} : Error {traceback.format_exc()}'


if __name__=='__main__':

    FS_HOME = os.environ.get('FREESURFER_HOME')
    subj_dir = settings.data_dir + 'freesurfer'
    os.makedirs(subj_dir, exist_ok=True)
    os.environ['SUBJECTS_DIR'] = subj_dir

    dataset = 'seq12'

    assert FS_HOME, '$FREESURFER_HOME not found in env'
    assert os.path.isfile(FS_HOME + '/bin/recon-all'), 'recon-all not found'
    assert FS_HOME in os.environ['PATH'], 'freesurfer not on $PATH'

    folders = ospath.list_folders(settings.data_dir + dataset, pattern='DSMR*')

    stop
    errs_recon = Parallel(30, backend='threading')(delayed(recon_all)(folder) for folder in tqdm(folders))
    errs_src = Parallel(30)(delayed(compute_source_space)(folder) for folder in tqdm(folders))
    errs_bem = Parallel(30)(delayed(compute_bem_solution)(folder) for folder in tqdm(folders))
    errs_cor = Parallel(30)(delayed(perform_coregistration)(folder) for folder in tqdm(folders))
    errs_cor = Parallel(30)(delayed(compute_forward)(folder) for folder in tqdm(folders))
