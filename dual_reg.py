from multiprocessing import Pool
import nibabel as nib
import pandas as pd
import numpy as np
import os

# setup global vars
condition = ['MOVIE2', 'MOVIE4', 'REST1', 'REST4']
hcp_path = '/arc/project/st-tv01-1/hcp'
data_path = os.path.join(hcp_path, 'data-clean')
out_path = '/scratch/st-tv01-1/hcp/targets/dual_reg'
indiv_out = os.path.join(out_path, 'indiv')
sub = np.loadtxt(os.path.join(hcp_path,'targets','m2m4_sub_n109.csv'), dtype=str)
dlpfc = '/arc/project/st-tv01-1/hcp/code/hallee_hcp_targets/data/dlpfc.dscalar.nii'
lh = '/arc/project/st-tv01-1/atlas/HCP_S1200/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
rh = '/arc/project/st-tv01-1/atlas/HCP_S1200/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii'

def run_clustering(thresh_prop):
    # collect sub, etc.
    df = pd.DataFrame(columns=['sub','condition','thresh_prop'])
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    for s in sub:
        for c in condition:
            df.loc[len(df)+1] = [s, c, thresh_prop]
    # run
    pool = Pool(cores)
    outputs = pool.starmap(cifti_cluster, df.values.tolist())
    pool.close()
    return outputs


def cifti_cluster(subject, cond, thresh_prop):
    dlpfc_mask = nib.load(dlpfc).get_fdata()[:,hcp.struct.cortex].squeeze() != 0
    cifti = f'{indiv_out}/{cond}_{subject}_seedmap.dscalar.nii'
    data = nib.load(cifti).get_fdata()[:,hcp.struct.cortex]
    for t in thresh_prop:
        cifti_out = f'{indiv_out}/{cond}_{subject}_seedmap-thresh{t}-cluster.dscalar.nii'
        thresh_dlpfc = np.quantile(data[:,dlpfc_mask], t)
        os.system(f'wb_command -cifti-find-clusters {cifti} {thresh_dlpfc} 0 0 0 COLUMN {cifti_out} -less-than -left-surface {lh} -right-surface {rh} -cifti-roi {dlpfc}')
