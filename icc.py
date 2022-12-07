import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sb
import sklearn as skl
import matplotlib as plot
import matplotlib.pyplot as plt
import scipy.stats as stats
import multiprocessing as mp
import pingouin as pg

def rearrange(sub, roi_name, inDir='/scratch/st-tv01-1/hcp/targets'):
    df = pd.DataFrame(columns = ['subject', 'run', 'value', 'edge', 'roi'])
    conditions = ['REST1', 'REST4', 'MOVIE2', 'MOVIE4']
    for cond in conditions:
        file = f'{inDir}/sub{sub}_{cond}_{roi_name}.csv'
        #print(file)
        matrix = pd.read_csv(file, sep=',', header=None)
        shape = matrix.values.flatten().shape[0]
        subject = np.full((shape,1), sub)
        run = np.full((shape,1), cond)
        values = np.transpose(np.array(matrix.values.flatten(), ndmin = 2))
        edge = np.transpose(np.array(range(0, shape), ndmin = 2))
        roi = np.full((shape,1), roi_name)
        df_to_add = pd.DataFrame(np.concatenate((subject, run, values, edge, roi), axis = 1), columns = ['subject', 'run', 'value', 'edge', 'roi'])
        df = pd.concat([df, df_to_add])
    return df

def icc_analysis(df):
    results = np.zeros((len(np.unique(df['edge']))*2, 3))
    count = 0
    for cond in ['REST', 'MOVIE']:
            df_run = df[df['run'].str.contains(cond)]
            for edge in np.unique(df['edge']):
                df_edge = df_run[df_run['edge']==edge]
                pg.intraclass_corr(data = df_edge, targets = 'subject', raters = 'run', ratings = 'value')
                results[count,0] = edge
                results[count,1] = icc.ICC[4]
                results[count,2] = cond
                count = count + 1
    return results

roi = ['dlpfc', 'tpj', 'pre_sma']
sub = pd.read_csv('/arc/project/st-tv01-1/hcp/targets/m2m4_sub_n111.csv', header = None).squeeze()

# loop over each roi
pool = mp.Pool(mp.cpu_count())
for r in roi:
    # setup starmap args
    args = []
    for s in sub:
        args.append((s, r))
    results = pool.starmap(rearrange, args)
    # concat subs to 1 df
    df = results.pop(0)
    while len(results) > 0:
        df = pd.concat((df, results.pop()))
    # run icc code and save
    icc = icc_analysis(df)
    del(df)
    np.savetxt(f'/scratch/st-tv01-1/hcp/targets/icc_{r}_n{len(sub)}.txt', icc)
    

pool.close()
print(df.shape)
print(df.head())
