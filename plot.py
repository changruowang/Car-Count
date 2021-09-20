'''
Author: your name
Date: 2021-09-19 13:08:46
LastEditTime: 2021-09-19 13:08:47
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \car_conut\plot.py
'''
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

if len(sys.argv) >= 3:
    d = pd.DataFrame.from_csv(sys.argv[1], index_col=None)
    d['5 seconds'] = (d['time']/(int(sys.argv[2])*100)).astype(int)
    d = d.groupby('5 seconds').agg({'car':np.sum,'truck':np.sum,'bus':np.sum,"time":lambda x: x.iloc[0]})
    d['time'] = (d['time']/100).astype(int)
    d['time'] = pd.to_datetime(d['time'],unit='s')
    d = d.set_index(['time'], drop=True)
    d.plot()
    plt.show()
else:
    print("Usage: python plot.py [path to the csv report] [number of seconds to aggregate by]")