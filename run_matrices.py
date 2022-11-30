from create_matrices import * # gives access to all functions in py file
import multiprocessing as mp
import os

repoDir = os.path.dirname(os.path.abspath(__file__))
#subject_list_small = [105923, 104416]
subject_list_n111 = pd.read_csv('/arc/project/st-tv01-1/hcp/targets/m2m4_sub_n111.csv', header = None).squeeze()
dataDir='/arc/project/st-tv01-1/hcp/data-clean'
saveDir='/scratch/st-tv01-1/hcp/targets'
# load Schaefer parcellation, remove medial wall
schaefer = nb.cifti2.load(os.path.join(repoDir,'data/schaefer2018/Schaefer2018_1000Parcels_7Networks_order.dscalar.nii')).get_fdata().squeeze()
medial = nb.cifti2.load(os.path.join(repoDir,'data/Human.MedialWall_Conte69.32k_fs_LR.dlabel.nii')).get_fdata().squeeze()
schaefer = schaefer[medial == 0]

args = []
for s in subject_list_n111:
    args.append((s, schaefer, dataDir, saveDir))
    
pool = mp.Pool(mp.cpu_count())
results = pool.starmap(get_all_matrices, args)
pool.close()
