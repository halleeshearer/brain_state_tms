from multiprocessing import Pool
import nibabel as nib
import pandas as pd
import numpy as np
import os
import hcp_utils as hcp
from glob import glob

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

def run_clustering(thresh_prop, cores=4):
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

def cluster_analyze(cores=4):
    # get thresholds run
    cifti = glob(f'{indiv_out}/*_seedmap-thresh*-cluster.dscalar.nii')
    thresh = []
    for c in cifti:
        thresh.append(os.path.basename(c).split('-')[1].replace('thresh',''))
    thresh = set(thresh)
    # find thresholds without outputs
    thresh_run = []
    for t in thresh:
        if (not os.path.isfile(f'{out_path}/seedmap_dlpfc_centroids_thresh{t}_n{len(sub)}.csv') or
            not os.path.isfile(f'{out_path}/intrasub_centroid_distances_thresh{t}_n{len(sub)}.csv')):
            thresh_run.append(t)
            print(f'Analyzing thresh={t}')
    # run
    pool = Pool(cores)
    outputs = pool.map(cluster_centroids, thresh_run)
    pool.close()
    return outputs

def cluster_centroids(thresh):
    dlpfc_dist = np.load(f'{out_path}/dlpfc_geodesic.npy')
    dlpfc_mask = nib.load(dlpfc).get_fdata()[:,hcp.struct.cortex_left].squeeze()
    dlpfc_vert = np.loadtxt(f'{out_path}/dlpfc_verts.txt')
    # find centroids
    df_cent = pd.DataFrame(columns=['sub', 'condition', 'cluster_size','centroid_vertex','centroid_dlpfc_index', 'centroid_value'])
    for i,s in enumerate(sub):
        for j,c in enumerate(condition):
            seed_data = nib.load(f'{indiv_out}/{c}_{s}_seedmap.dscalar.nii').get_fdata()[:,hcp.struct.cortex_left].squeeze()
            clust_data = nib.load(f'{indiv_out}/{c}_{s}_seedmap-thresh{thresh}-cluster.dscalar.nii').get_fdata()[:,hcp.struct.cortex_left].squeeze()
            # restrict to dlpfc
            clust_data = clust_data[dlpfc_mask != 0]
            seed_data = seed_data[dlpfc_mask != 0]
            # get cluster sizes
            clust_num = np.unique(clust_data)
            clust_num = clust_num[clust_num!=0]
            clust_size = np.zeros(clust_num.shape)
            for k,n in enumerate(np.unique(clust_num)):
                if n != 0:
                    clust_size[k] = sum(clust_data == n)
            # find largest clust
            clust_num_largest = clust_num[clust_size.argmax()]
            clust_mask = clust_data == clust_num_largest
            # find centroid
            clust_dist = dlpfc_dist[clust_mask, :]
            clust_dist = clust_dist[:, clust_mask].sum(axis=0)
            cent_idx = np.where(clust_mask)[0][clust_dist.argmin()]
            cent_vert = dlpfc_vert[cent_idx]
            cent_val = seed_data[cent_idx]
            df_cent.loc[df_cent.shape[0]+1,:] = [s, c, clust_size.max(), cent_vert, cent_idx, cent_val]
    df_cent.to_csv(f'{out_path}/seedmap_dlpfc_centroids_thresh{thresh}_n{len(sub)}.csv', index=False)

    # compute distances
    df_dist = pd.DataFrame(columns=['sub', 'condition_pair', 'centroid_distance'])
    for i,s in enumerate(sub):
        cond_idx = df_cent.loc[df_cent['sub'] ==int(s),['condition', 'centroid_dlpfc_index']].set_index('condition')
        cond_idx = cond_idx.loc[condition,'centroid_dlpfc_index'].tolist()
        cond_dist = dlpfc_dist[cond_idx, :]
        cond_dist = cond_dist[:, cond_idx]
        for j1,c1 in enumerate(condition):
            for j2,c2 in enumerate(condition):
                if j1<j2:
                    df_dist.loc[df_dist.shape[0]+1,:] = [s, f'{c1}_{c2}', cond_dist[j1,j2]]
    df_dist = df_dist.pivot(index='sub', columns='condition_pair')
    df_dist.to_csv(f'{out_path}/intrasub_centroid_distances_thresh{thresh}_n{len(sub)}.csv')

    return (f'{out_path}/seedmap_dlpfc_centroids_thresh{thresh}_n{len(sub)}.csv',f'{out_path}/intrasub_centroid_distances_thresh{thresh}_n{len(sub)}.csv')
