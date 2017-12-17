#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 09:44:15 2017

@author: konodera
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from collections import Counter

n_children = 1000000 # n children to give
n_gift_type = 1000 # n types of gifts available
n_gift_quantity = 1000 # each type of gifts are limited to this quantity
n_gift_pref = 10 # number of gifts a child ranks
n_child_pref = 1000 # number of children a gift ranks
twins = int(0.004 * n_children)    # 0.4% of all population, rounded to the closest even number
ratio_gift_happiness = 2
ratio_child_happiness = 2


gift_pref = pd.read_csv('../input/child_wishlist.csv.zip',header=None).drop(0, 1).values
child_pref = pd.read_csv('../input/gift_goodkids.csv.zip',header=None).drop(0, 1).values

# input file
pred_start = pd.read_csv('../output/sub3765342892.csv.gz').values.tolist()

# output file
outfile = '../output/sub3765342892-twin_hosei.csv.gz'





# =============================================================================
# # modify for twins
# =============================================================================

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


Gifts_list = [[] for i in range(1000)]
for p in pred_start:
    Gifts_list[p[1]].append(p[0])


Gifts_list_2 = Gifts_list.copy()

res_list_temp = [[] for i in range(1000000)]
for j in range(1000):
    for i in Gifts_list[j]:
        res_list_temp[i] = [i, j]


gain_move = 0
for i in range(2000):
    k1 = res_list_temp[2*i][1]
    k2 = res_list_temp[2*i+1][1]
    # which to go
    if k1 == k2:
        pass
    else:
        # 2*i move
        gain1 = Children[2*i].happiness(k2) - Children[2*i].happiness(k1)
        gain1_add = -10000000
        for l in Gifts_list_2[k2]:
            gain1_add_ = Children[l].happiness(k1) - Children[l].happiness(k2)
            if gain1_add_ > gain1_add and l > 2*i+1:
                v1 = l
                gain1_add = gain1_add_
        # 2*i+1 move
        gain2 = Children[2*i].happiness(k1) - Children[2*i].happiness(k2)
        gain2_add = -10000000
        for l in Gifts_list_2[k1]:
            gain2_add_ = Children[l].happiness(k2) - Children[l].happiness(k1)
            if gain2_add_ > gain2_add and l > 2*i+1:
                v2 = l
                gain2_add = gain2_add_
                
        if gain1 + gain1_add >= gain2 + gain2_add:
            res_list_temp[2*i][1] = k2
            res_list_temp[v1][1] = k1
            Gifts_list_2[k1].remove(2*i)
            Gifts_list_2[k2].append(2*i)
            Gifts_list_2[k2].remove(v1)
            Gifts_list_2[k1].append(v1)
        else:
            res_list_temp[2*i+1][1] = k1
            res_list_temp[v2][1] = k2
            Gifts_list_2[k2].remove(2*i+1)
            Gifts_list_2[k1].append(2*i+1)
            Gifts_list_2[k1].remove(v2)
            Gifts_list_2[k2].append(v2)
        gain_move += max(gain1 + gain1_add, gain2 + gain2_add)
print(gain_move)


df = pd.DataFrame(res_list_temp,
                  columns=['ChildId','GiftId'])
df.to_csv(outfile, index=False, compression='gzip')

#out = open(outfile, 'w')
#out.write('ChildId,GiftId\n')
#for i in range(len(res_list_temp)):
#    out.write(str(i) + ',' + str(res_list_temp[i][1]) + '\n')
#out.close()


