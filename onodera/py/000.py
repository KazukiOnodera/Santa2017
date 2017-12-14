#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 08:54:21 2017

@author: konodera

nohup python -u 000.py > log.txt &


"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import utils
#utils.start(__file__)

seed = 71
total_proc = 40

np.random.seed(seed)

sub = pd.read_csv('../output/sub1213-1.csv.gz')

children = utils.Children(sub.ChildId.values, sub.GiftId.values)

print("total_happiness:", children.total_happiness())

cids_twins     = np.arange(0, 4000)
cids_not_twins = np.arange(4000, 1000000)

# =============================================================================
# 
# =============================================================================
def multi_1on1(args):
    return children.eval_gift_1on1(args[0], args[1])

def multi_3(args):
    return children.eval_gift_3(args[0], args[1], args[2])

def multi_twins(args):
    return children.eval_gift_twins(args[0], args[1])


for i in range(99999999):
    
    # 1on1
    np.random.shuffle(cids_not_twins)
    cids_not_twins_ = list(zip(*[iter(cids_not_twins)]*2))
    callback = Parallel(n_jobs=total_proc)( [delayed(multi_1on1)(cids) for cids in cids_not_twins_] )
    index = np.where(np.array(callback)>0)[0]
    
    for j in index:
        cid1, cid2 = cids_not_twins_[j]
        score = children.eval_gift_1on1(cid1, cid2)
        if score >= 0:
            print('Accept!!', score)
            children[cid1].accept()
            children[cid2].accept()
    
    # 3
    np.random.shuffle(cids_not_twins)
    cids_not_twins_ = list(zip(*[iter(cids_not_twins)]*3))
    callback = Parallel(n_jobs=total_proc)( [delayed(multi_3)(cids) for cids in cids_not_twins_] )
    index = np.where(np.array(callback)>0)[0]
    
    for j in index:
        cid1, cid2, cid3 = cids_not_twins_[j]
        score = children.eval_gift_3(cid1, cid2, cid3)
        if score >= 0:
            print('Accept!!', score)
            children[cid1].accept()
            children[cid2].accept()
            children[cid3].accept()
    
    # twins vs twins
    np.random.shuffle(cids_twins)
    cids_twins_ = list(zip(*[iter(cids_twins)]*2))
    callback = Parallel(n_jobs=total_proc)( [delayed(multi_twins)(cids) for cids in cids_twins_] )
    index = np.where(np.array(callback)>0)[0]
    
    for j in index:
        cid1, cid2 = cids_twins_[j]
        score = children.eval_gift_twins(cid1, cid2)
        if score >= 0:
            print('Accept!!', score)
            children[cid1].accept()
            children[cid2].accept()
    
    if i%3==0:
        total_hp = children.total_happiness()
        print("total_happiness:", total_hp)
        children.mk_sub('../output/sub{}.csv.gz'.format(total_hp))



#utils.end(__file__)

