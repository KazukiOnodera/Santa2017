#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 08:54:21 2017

@author: konodera

nohup python -u 002.py > log.txt &

最初のループではchild Aを全部舐めて、
2番目はchild Aの好みを舐めて、
3番目ではそのAの好みを持ってるchildを探す

"""

import pandas as pd
import numpy as np
from operator import itemgetter
import time
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from multiprocessing import Pool
import utils
#utils.start(__file__)

seed = 71
total_proc = 40
timelimit = 60*60*4

input_file  = '../output/sub941172.7.csv.gz'
#output_file = '../output/subm_ond1216_child-vs-child.csv.gz'

np.random.seed(seed)
# =============================================================================
# preprocess
# =============================================================================
n_children = 1000000 # n children to give
n_gift_type = 1000 # n types of gifts available
n_gift_quantity = 1000 # each type of gifts are limited to this quantity
n_gift_pref = 10 # number of gifts a child ranks
n_child_pref = 1000 # number of children a gift ranks
twins = int(0.004 * n_children)    # 0.4% of all population, rounded to the closest even number
ratio_gift_happiness = 2
ratio_child_happiness = 2

#child = pd.read_csv('../input/child_wishlist.csv.zip', header=None)
#child.columns = ['cid'] + list(range(1, child.shape[1]))
#
#gift = pd.read_csv('../input/gift_goodkids.csv.zip', header=None)
#gift.columns = ['gid'] + list(range(1, gift.shape[1]))


#def get_twins_id(id):
#    if id<=3999:
#        if id%2==0:
#            return id+1
#        else:
#            return id-1
#    else:
#        return -1
#
#def total_happiness(children, gifts):
#    [c.set_happiness() for c in children]
#    ret = np.sum([c.happiness for c in children])
#    [g.set_happiness() for g in gifts]
#    ret += np.sum([g.happiness for g in gifts])
#    return ret
#
#def happiness_diff(children, gifts, cid, gid1, gid2):
#    d  = children[cid].get_happiness(gid1) - children[cid].get_happiness(gid2)
#    d += gifts[gid1].get_happiness(cid) - gifts[gid2].get_happiness(cid)
#    return d
#
#class Children:
#    """
#    cid is ChildId
#    gid is GiftId
#    """
#    def __init__(self, cids, gids):
#        self.children = []
#        self.gifts = defaultdict(list)
#        for cid, gid in zip(tqdm(cids, miniters=9999), gids):
#            self.children.append(Child(cid, gid))
#            self.gifts[gid].append(cid)
#    
#    def __getitem__(self, index):
#        return self.children[index]
#    
#    def __len__(self):
#        return len(self.children)
#    
#    def replace(self, cid1, cid2):
#        gid1 = self[cid1].gid
#        gid2 = self[cid2].gid
#        self[cid1].gid = gid2
#        self[cid2].gid = gid1
#        
#    def mk_sub(self, path):
#        idset = [(c.id,c.gid) for c in self.children]
#        sub = pd.DataFrame(idset,
#                           columns=['ChildId', 'GiftId'])
#        sub.to_csv(path, index=False, compression='gzip')
#        return
#
#def get_child(cid):
#    return child.iloc[cid].values
#
#class Child:
#    """
#    """
#    def __init__(self, cid, gid):
#        values = get_child(cid)
#        self.id   = values[0]
#        self.twins_id = get_twins_id(self.id)
#        self.pref = values[1:]
#        self.happiness = -1/20
#        self.gid = gid
#    
#    def get_happiness(self, gid):
#        try:
#            hp = (n_gift_pref - np.where(self.pref==gid)[0][0]) * ratio_child_happiness
#        except IndexError:
#            hp = -1
#        hp /=20.
#        return hp
#    
#    def set_gid(self, gid):
#        self.gid = gid
#    
#    def set_happiness(self):
#        self.happiness = self.get_happiness(self.gid)
#
#def get_gift(gid):
#    return gift.iloc[gid].values
#
#class Gifts:
#    """
#    """
#    def __init__(self, gids, cids):
#        gifts = defaultdict(list)
#        for cid, gid in zip(cids, gids):
#            gifts[gid].append(cid)
#        self.gifts = []
#        for k,v in gifts.items():
#            self.gifts.append(Gift(k,v))
#    
#    def __getitem__(self, index):
#        return self.gifts[index]
#    
#    def __len__(self):
#        return len(self.children)
#    
#    def replace(self, gid_remove, gid_append, cid):
#        self[gid_remove].remove_cid(cid)
#        self[gid_append].append_cid(cid)
#        
#class Gift:
#    """
#    """
#    def __init__(self, gid, cids):
#        values = get_gift(gid)
#        self.id = values[0]
#        self.pref = values[1:]
#        self.cids = cids
#    
#    def get_happiness(self, cid):
#        try:
#            hp = (n_child_pref - np.where(self.pref==cid)[0][0]) * ratio_gift_happiness
#        except IndexError:
#            hp = -1
#        hp /=2000.
#        return hp
#    
#    def remove_cid(self, cid):
#        if cid in self.cids:
#            self.cids.remove(cid)
#            
#    def append_cid(self, cid):
#        if cid not in self.cids:
#            self.cids.append(cid)
#    
#    def set_happiness(self):
#        self.happiness = 0
#        self.happiness += np.sum([self.get_happiness(cid) for cid in self.cids])


class Child(object):
    
    def __init__(self, idx, prefer):
        
        self.idx = idx
        self.prefer_dict = dict()
        
        for i in range(prefer.shape[0]):
            self.prefer_dict[prefer[i]] = 400*(prefer.shape[0] - i) - 2
    
    
    def add_gifts_prefer(self, giftid, score):
        
        if giftid in self.prefer_dict.keys():
            self.prefer_dict[giftid] += 2*score + 2
        else:
            self.prefer_dict[giftid] = 2*score - 200
        
        return None
        
    
    def happiness(self, giftid):
        
        return self.prefer_dict.get(giftid, -202)


class Child_twin(object):
    
    def __init__(self, idx, prefer1, prefer2):
        
        self.idx = idx
        self.prefer_dict = dict()
        
        for p in list(set(list(prefer1) + list(prefer2))):
            score = 0
            if p in list(prefer1):
                score += 2*(10 - list(prefer1).index(p))
            else:
                score -= 1
            if p in list(prefer2):
                score += 2*(10 - list(prefer2).index(p))
            else:
                score -= 1
            self.prefer_dict[p] = 100*score - 2
    
    
    def add_gifts_prefer(self, giftid, score):
        
        if giftid in self.prefer_dict.keys():
            self.prefer_dict[giftid] += score + 2
        else:
            self.prefer_dict[giftid] = score - 200
        
        return None
        
    
    def happiness(self, giftid):
        
        return self.prefer_dict.get(giftid, -202)



gift_pref = pd.read_csv('../input/child_wishlist.csv.zip',header=None).drop(0, 1).values
child_pref = pd.read_csv('../input/gift_goodkids.csv.zip',header=None).drop(0, 1).values

Children = []
for i in range(2000):
    Children.append(Child_twin(2*i, gift_pref[2*i], gift_pref[2*i+1]))
    Children.append(Child_twin(2*i+1, gift_pref[2*i], gift_pref[2*i+1]))
Children = Children + [Child(i, gift_pref[i]) for i in range(4000, 1000000)]


for j in range(1000):
    cf = child_pref[j]
    done_list = []
    for i in range(cf.shape[0]):
        if cf[i] < 4000 and cf[i] not in done_list:
            if cf[i] % 2 == 0:
                cid1 = cf[i]
                cid2 = cf[i] + 1
                done_list.append(cid2)
            else:
                cid1 = cf[i] - 1
                cid2 = cf[i]
                done_list.append(cid1)
            if cid1 in list(cf):
                score_ = 2*(cf.shape[0] - list(cf).index(cid1))
            else:
                score_ = -1
            if cid2 in list(cf):
                score_ += 2*(cf.shape[0] - list(cf).index(cid2))
            else:
                score_ += -1
            Children[cid1].add_gifts_prefer(j, score_)
            Children[cid2].add_gifts_prefer(j, score_)
        elif cf[i] >= 4000:
            Children[cf[i]].add_gifts_prefer(j, 2*(cf.shape[0] - i))

pred_start = pd.read_csv(input_file).values.tolist()

Gifts_list = [[] for i in range(1000)]
for p in pred_start:
    Gifts_list[p[1]].append(p[0])





#sub = pd.read_csv(input_file)
#
#children = Children(sub.ChildId.values, sub.GiftId.values)
#gifts    = Gifts(sub.GiftId.values, sub.ChildId.values)

#print("total_happiness:", utils.total_happiness(children, gifts))

cids_twins     = np.arange(0, 4000)
cids_not_twins = np.arange(4000, 1000000)
cids = np.arange(0, 1000000)

# =============================================================================
# main
# =============================================================================

#def child_vs_child(cid1):
#    ret = []
#    for gidA in children[cid1].pref:
#        if gidA==children[cid1].gid:
#            break
#        for cid2 in gifts[gidA].cids:
#            if cid2<4000:
#                continue
#            gidB = children[cid1].gid
#            d = happiness_diff(children, gifts, cid1, gidA, gidB) + happiness_diff(children, gifts, cid2, gidB, gidA)
#            if d>=0:
#                ret.append((cid1, cid2, d))
#    return ret

def happiness_diff(cid, gid1, gid2):
    d = Children[cid].happiness(gid1) - Children[cid].happiness(gid2)
    return d

def child_vs_child2(cid1):
    ret = []
    gid1 = Gifts_list
    cur_hp = Children[cid].happiness()
    for gidA in children[cid1].pref:
        if gidA==children[cid1].gid:
            break
        for cid2 in gifts[gidA].cids:
            if cid2<4000:
                continue
            gidB = children[cid1].gid
            d = happiness_diff(cid1, gidA, gidB) + happiness_diff(cid2, gidB, gidA)
            if d>=0:
                ret.append((cid1, cid2, d))
    return ret


cnt = 0
st_time = time.time()

while True:
    cnt +=1
    
    # mp child_vs_child
    pool = Pool(total_proc)
    callback = pool.map(child_vs_child, np.random.choice(cids_not_twins, replace=False, size=10000))
    pool.close()
    callback = sum(callback, [])
    callback = sorted(callback, key=itemgetter(2), reverse=True)
    
    ng_list = []
    for (cid1, cid2, d) in callback:
        if cid1 in ng_list or cid2 in ng_list or cid1<4000 or cid2<4000:
            continue
        ng_list.append(cid1); ng_list.append(cid2)
        gid1 = children[cid1].gid
        gid2 = children[cid2].gid
        children.replace(cid1, cid2)
        gifts.replace(gid1, gid2, cid1)
        gifts.replace(gid2, gid1, cid2)
    
    if cnt%10==0:
        d = time.time()-st_time
        total_hp = utils.total_happiness(children, gifts)
        print("cnt = {} : total_happiness = {} : elaped = {}".format(cnt, total_hp, d))
        children.mk_sub('../output/sub{}.csv.gz'.format(total_hp))
        if d>timelimit:
            break


utils.end(__file__)

