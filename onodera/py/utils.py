#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:20:40 2017

@author: konodera
"""
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
from glob import glob
from collections import defaultdict
#import pickle
#import config

def start(fname):
    global st_time
    st_time = time.time()
    print("""
#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(fname, os.getpid()))
    
    return

def end(fname):
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format( (time.time() - st_time)/60 ))
    return

def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)
    
def to_pickles(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'
    
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    
    """
    
    if inplace==True:
        df.reset_index(drop=True, inplace=inplace)
    else:
        df = df.reset_index(drop=True)
        
    mkdir_p(path)
    
    for i in tqdm(range(split_size)):
        df.ix[df.index%split_size==i].to_pickle(path+'/{}.p'.format(i))
    
    return

def read_pickles(path, col=None, how='and'):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    elif col is not None:
        if how=='and':
            df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
        elif how=='safe':
            """
            df.columns
            >>> [c1, c2, c3]
            col
            >>> [c1, c2, c4]
            return:
                df[[c1, c2]]
            """
            col_p = pd.read_pickle(path+'/0.p').columns
            col = [c for c in col if c in col_p]
            df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
        else:
            raise Exception('Invalid how:', how)
    return df

# =============================================================================
# for Santa2017
# =============================================================================
n_children = 1000000 # n children to give
n_gift_type = 1000 # n types of gifts available
n_gift_quantity = 1000 # each type of gifts are limited to this quantity
n_gift_pref = 10 # number of gifts a child ranks
n_child_pref = 1000 # number of children a gift ranks
twins = int(0.004 * n_children)    # 0.4% of all population, rounded to the closest even number
ratio_gift_happiness = 2
ratio_child_happiness = 2

child = pd.read_csv('../input/child_wishlist.csv.zip', header=None)
child.columns = ['cid'] + list(range(1, child.shape[1]))

gift = pd.read_csv('../input/gift_goodkids.csv.zip', header=None)
gift.columns = ['gid'] + list(range(1, gift.shape[1]))


def get_twins_id(id):
    if id<=3999:
        if id%2==0:
            return id+1
        else:
            return id-1
    else:
        return -1

def total_happiness(children, gifts):
    [c.set_happiness() for c in children]
    ret = np.sum([c.happiness for c in children])
    [g.set_happiness() for g in gifts]
    ret += np.sum([g.happiness for g in gifts])
    return ret

def happiness_diff(children, gifts, cid, gid1, gid2):
    d  = children[cid].get_happiness(gid1) - children[cid].get_happiness(gid2)
    d += gifts[gid1].get_happiness(cid) - gifts[gid2].get_happiness(cid)
    return d

def swap_gift_1on1_update(children, gifts, gid1, gid2, gid1_list, gid2_list):
    # child
    [children.replace(cid, gid2) for cid in gid2_list]
    [children.replace(cid, gid1) for cid in gid1_list]
    
    # gift
    [gifts.replace(gid1, gid2, cid) for cid in gid2_list]
    [gifts.replace(gid2, gid1, cid) for cid in gid1_list]
    return

def swap_gift_1on1(children, gifts, gid1, gid2):
    """
    gid1 -> gid2
    """
    
    target_children = gifts[gid1].cids + gifts[gid2].cids
    prefer_order = np.argsort([happiness_diff(children, gifts, c, gid1, gid2) for c in target_children])
    
    gid2_list = np.array([target_children[k] for k in prefer_order[:1000]])
    gid1_list = np.array([target_children[k] for k in prefer_order[1000:]])
    
    gid2_twins = np.where(gid2_list<4000)[0]
    gid1_twins = np.where(gid1_list<4000)[0]
#    gid2_twins//2
    
    # no twins
    if len(gid2_twins)==0 or len(gid1_twins)==0:
        gid1_list = set(gid1_list) - set(gifts[gid1].cids)
        gid2_list = set(gid2_list) - set(gifts[gid2].cids)
        if len(gid1_list)>0 and len(gid2_list)>0:
            return gid1, gid2, gid1_list, gid2_list
        
    return 

class Children:
    """
    cid is ChildId
    gid is GiftId
    """
    def __init__(self, cids, gids):
        self.children = []
        self.gifts = defaultdict(list)
        for cid, gid in zip(tqdm(cids, miniters=9999), gids):
            self.children.append(Child(cid, gid))
            self.gifts[gid].append(cid)
    
    def __getitem__(self, index):
        return self.children[index]
    
    def __len__(self):
        return len(self.children)
    
    def replace(self, cid, gid):
        self[cid].gid = gid
        self[cid].set_happiness()
        
    def mk_sub(self, path):
        idset = [(c.id,c.gid) for c in self.children]
        sub = pd.DataFrame(idset,
                           columns=['ChildId', 'GiftId'])
        sub.to_csv(path, index=False, compression='gzip')
        return

def get_child(cid):
    return child.iloc[cid].values

class Child:
    """
    """
    def __init__(self, cid, gid):
        values = get_child(cid)
        self.id   = values[0]
        self.twins_id = get_twins_id(self.id)
        self.pref = values[1:]
        self.happiness = -1/20
        self.gid = gid
    
    def get_happiness(self, gid):
        try:
            hp = (n_gift_pref - np.where(self.pref==gid)[0][0]) * ratio_child_happiness
        except IndexError:
            hp = -1
        hp /=20.
        return hp
    
    def set_gid(self, gid):
        self.gid = gid
    
    def set_happiness(self):
        self.happiness = self.get_happiness(self.gid)

def get_gift(gid):
    return gift.iloc[gid].values

class Gifts:
    """
    """
    def __init__(self, gids, cids):
        gifts = defaultdict(list)
        for cid, gid in zip(cids, gids):
            gifts[gid].append(cid)
        self.gifts = []
        for k,v in gifts.items():
            self.gifts.append(Gift(k,v))
    
    def __getitem__(self, index):
        return self.gifts[index]
    
    def __len__(self):
        return len(self.children)
    
    def replace(self, gid_remove, gid_append, cid):
        self[gid_remove].remove_cid(cid)
        self[gid_append].append_cid(cid)
        
class Gift:
    """
    """
    def __init__(self, gid, cids):
        values = get_gift(gid)
        self.id = values[0]
        self.pref = values[1:]
        self.cids = cids
    
    def get_happiness(self, cid):
        try:
            hp = (n_child_pref - np.where(self.pref==cid)[0][0]) * ratio_gift_happiness
        except IndexError:
            hp = -1
        hp /=2000.
        return hp
    
    def remove_cid(self, cid):
        if cid in self.cids:
            self.cids.remove(cid)
            
    def append_cid(self, cid):
        if cid not in self.cids:
            self.cids.append(cid)
    
    def set_happiness(self):
        self.happiness = 0
        self.happiness += np.sum([self.get_happiness(cid) for cid in self.cids])
        
    
    
    
    
    
    