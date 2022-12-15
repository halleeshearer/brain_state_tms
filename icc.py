import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sb
import sklearn as skl
import matplotlib as plot
import matplotlib.pyplot as plt
import scipy.stats as stats

import pingouin as pg

import argparse
import sys
import time
import multiprocessing as mp

def rearrange(sub, roi_name, conditions, inDir='/scratch/st-tv01-1/hcp/targets'):
    df = pd.DataFrame(columns = ['subject', 'run', 'value', 'edge', 'roi'])
    for cond in conditions:
        print(f'loading {sub} {cond}...')
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

def icc(df):
    if df[0] % 50000 == 0:
        print(f'edge {df[0]}')
    return (df[0], pg.intraclass_corr(data = df[1], targets = 'subject', raters = 'run', ratings = 'value').ICC[4])
    

def main(logfile=None):
    if logfile is not None:
        f = open(logfile, 'w')
        sys.stdout = f
        
    roi = ['dlpfc', 'tpj', 'pre_sma']
    conditions = ['REST1', 'REST4', 'MOVIE2', 'MOVIE4']
    sub = pd.read_csv('/arc/project/st-tv01-1/hcp/targets/m2m4_sub_n111.csv', header = None).squeeze()
    # loop over each roi
    pool = mp.Pool(mp.cpu_count())
    for r in roi:
        for c in ['REST','MOVIE']:
            # setup starmap args
            args = []
            start = time.time()
            for s in sub:
                #print(f'loading {s}...')
                args.append((s, r, [x for x sin conditions if x.startswith(c)]))
            results = pool.starmap(rearrange, args)
            # concat subs to 1 df
            df = results.pop(0)
            while len(results) > 0:
                df = pd.concat((df, results.pop()))
            # run icc code and save
            minutes = round((time.time() - start) / 60,1)
            print(f'all {r} {c} data loaded in {minutes} min')
            start = time.time()
            df = list(df.groupby('edge'))
            icc = pool.map(icc_analysis, df)
            minutes = round((time.time() - start)/60,1)
            print(f'{r} {c} calculated in {minutes}')
            del(df)
            np.savetxt(f'/scratch/st-tv01-1/hcp/targets/icc_{r}_n{len(sub)}.txt', np.array(icc))
    pool.close()
    print(df.shape)
    print(df.head())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", default=None)
    args = parser.parse_args()
    main(args.logfile)
