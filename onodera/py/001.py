#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 08:54:21 2017

@author: konodera

nohup python -u 001.py > log.txt &


"""

import pandas as pd
import numpy as np
#from joblib import Parallel, delayed
from multiprocessing import Pool
import utils
utils.start(__file__)

seed = 71
total_proc = 40

np.random.seed(seed)

sub = pd.read_csv('../output/6148153.zip')

children = utils.Children(sub.ChildId.values, sub.GiftId.values)
gifts    = utils.Gifts(sub.GiftId.values, sub.ChildId.values)

print("total_happiness:", utils.total_happiness(children, gifts))

cids_twins     = np.arange(0, 4000)
cids_not_twins = np.arange(4000, 1000000)
gids = np.arange(0, 1000)

# =============================================================================
# 
# =============================================================================
def multi_1on1(args):
    gid1, gid2 = args
    return utils.swap_gift_1on1(children, gifts, gid1, gid2)

pool = Pool(total_proc)

for i in range(99999999):
    
    # joblib 1on1
#    np.random.shuffle(gids)
#    gids_ = list(zip(*[iter(gids)]*2))
#    Parallel(n_jobs=total_proc)( [delayed(multi_1on1)(gg) for gg in gids_] )
    
    # mp 1on1
    np.random.shuffle(gids)
    gids_ = list(zip(*[iter(gids)]*2))
    callback = pool.map(multi_1on1, gids_)
    callback = [x for x in callback if x is not None]
    for x in callback:
        utils.swap_gift_1on1_update(children, gifts, x[0], x[1], x[2], x[3])
        
    # single 1on1
#    gid1, gid2 = np.random.choice(1000, 2, replace=False)
#    utils.swap_gift_1on1(children, gifts, gid1, gid2)
    
    if i%100==0:
        total_hp = utils.total_happiness(children, gifts)
        print("total_happiness:", total_hp)
        children.mk_sub('../output/sub{}.csv.gz'.format(total_hp))



utils.end(__file__)

