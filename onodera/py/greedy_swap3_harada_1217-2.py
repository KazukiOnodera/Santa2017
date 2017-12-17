
# coding: utf-8

# # 公式のkernel

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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
pred_start = pd.read_csv('../output/sub3765342892-twin_hosei.csv.gz').values.tolist()

# output file
outfile = '../output/subm_ond1217_swap3-2.csv'

timelimit = 60*60*6

seed = 73
np.random.seed(seed)

# In[3]:

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


# In[4]:

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


# In[5]:

Children = []
for i in range(2000):
    Children.append(Child_twin(2*i, gift_pref[2*i], gift_pref[2*i+1]))
    Children.append(Child_twin(2*i+1, gift_pref[2*i], gift_pref[2*i+1]))
Children = Children + [Child(i, gift_pref[i]) for i in range(4000, 1000000)]


# In[6]:

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


# # スタート  
# subm_hrd1214_10000000.csv

# In[7]:



# In[8]:

Gifts_list = [[] for i in range(1000)]
for p in pred_start:
    Gifts_list[p[1]].append(p[0])


# In[9]:

import heapq

def swap3_greedy_all_multi(ggg):
    i, j, k = ggg
    all_users = Gifts_list[i] + Gifts_list[j] + Gifts_list[k]
    
    # sort by happiness_i - happiness_j
    prefer_order = np.argsort([Children[c].happiness(i) - Children[c].happiness(j) for c in all_users])
    users_sorted = [all_users[l] for l in prefer_order]
    # former prefer j to i
    # former 1000+K(0 <= K <= 1000) children should get j or k
    
    left_hap = sum([Children[users_sorted[l]].happiness(j) for l in range(1000)])
    happiness_list = [left_hap]
    left_queue = [Children[users_sorted[l]].happiness(j) - Children[users_sorted[l]].happiness(k) for l in range(1000)]
    heapq.heapify(left_queue)
    for K in range(1000):
        heapq.heappush(left_queue, Children[users_sorted[1000+K]].happiness(j) - Children[users_sorted[1000+K]].happiness(k))
        left_hap += Children[users_sorted[1000+K]].happiness(j)
        left_hap -= heapq.heappop(left_queue)
        happiness_list.append(left_hap)
    
    # latter 1000+L
    right_hap = sum([Children[users_sorted[l]].happiness(i) for l in range(2000, 3000)])
    happiness_list[-1] += right_hap
    right_queue = [Children[users_sorted[l]].happiness(i) - Children[users_sorted[l]].happiness(k) for l in range(2000, 3000)]
    heapq.heapify(right_queue)
    for L in range(1000):
        heapq.heappush(right_queue, Children[users_sorted[1999-L]].happiness(i) - Children[users_sorted[1999-L]].happiness(k))
        right_hap += Children[users_sorted[1999-L]].happiness(i)
        right_hap -= heapq.heappop(right_queue)
        happiness_list[999-L] += right_hap
    
    # which.max
    K_best = happiness_list.index(max(happiness_list))
    # former (j or k)
    former_users = users_sorted[:(1000+K_best)]
    prefer_order_1 = np.argsort([Children[c].happiness(k) - Children[c].happiness(j) for c in former_users])
    users_sorted_1 = [former_users[l] for l in prefer_order_1]
    # latter (i or k)
    latter_users = users_sorted[(1000+K_best):]
    prefer_order_2 = np.argsort([Children[c].happiness(k) - Children[c].happiness(i) for c in latter_users])
    users_sorted_2 = [latter_users[l] for l in prefer_order_2]
    
    return i,j,k, users_sorted_2[:1000], users_sorted_1[:1000], users_sorted_2[1000:] + users_sorted_1[1000:]

def swap3_greedy_all(i, j, k):
    # i, j, k : item
    # get current children who are going to be presented gift i or j or k
    # returns : swaped list
    # ignore twins condition
    # See https://atcoder.jp/img/agc018/editorial.pdf (Problem C) for details
    
    all_users = Gifts_list[i] + Gifts_list[j] + Gifts_list[k]
    
    # sort by happiness_i - happiness_j
    prefer_order = np.argsort([Children[c].happiness(i) - Children[c].happiness(j) for c in all_users])
    users_sorted = [all_users[l] for l in prefer_order]
    # former prefer j to i
    # former 1000+K(0 <= K <= 1000) children should get j or k
    
    left_hap = sum([Children[users_sorted[l]].happiness(j) for l in range(1000)])
    happiness_list = [left_hap]
    left_queue = [Children[users_sorted[l]].happiness(j) - Children[users_sorted[l]].happiness(k) for l in range(1000)]
    heapq.heapify(left_queue)
    for K in range(1000):
        heapq.heappush(left_queue, Children[users_sorted[1000+K]].happiness(j) - Children[users_sorted[1000+K]].happiness(k))
        left_hap += Children[users_sorted[1000+K]].happiness(j)
        left_hap -= heapq.heappop(left_queue)
        happiness_list.append(left_hap)
    
    # latter 1000+L
    right_hap = sum([Children[users_sorted[l]].happiness(i) for l in range(2000, 3000)])
    happiness_list[-1] += right_hap
    right_queue = [Children[users_sorted[l]].happiness(i) - Children[users_sorted[l]].happiness(k) for l in range(2000, 3000)]
    heapq.heapify(right_queue)
    for L in range(1000):
        heapq.heappush(right_queue, Children[users_sorted[1999-L]].happiness(i) - Children[users_sorted[1999-L]].happiness(k))
        right_hap += Children[users_sorted[1999-L]].happiness(i)
        right_hap -= heapq.heappop(right_queue)
        happiness_list[999-L] += right_hap
    
    # which.max
    K_best = happiness_list.index(max(happiness_list))
    # former (j or k)
    former_users = users_sorted[:(1000+K_best)]
    prefer_order_1 = np.argsort([Children[c].happiness(k) - Children[c].happiness(j) for c in former_users])
    users_sorted_1 = [former_users[l] for l in prefer_order_1]
    # latter (i or k)
    latter_users = users_sorted[(1000+K_best):]
    prefer_order_2 = np.argsort([Children[c].happiness(k) - Children[c].happiness(i) for c in latter_users])
    users_sorted_2 = [latter_users[l] for l in prefer_order_2]
    
    return users_sorted_2[:1000], users_sorted_1[:1000], users_sorted_2[1000:] + users_sorted_1[1000:]


# In[17]:
# multi

from multiprocessing import Pool
total_proc = 40
gids = np.arange(0, 1000)

add_res = t = 0
st_time = time.time()

while True:
    t +=1
    
    pool = Pool(total_proc)
    np.random.shuffle(gids)
    gids_ = list(zip(*[iter(gids)]*3))
    callback = pool.map(swap3_greedy_all_multi, gids_)
    pool.close()
    
#    i, j, k = np.random.choice(1000, 3, replace=False)
#    i_list, j_list, k_list = swap3_greedy_all(i, j, k)
    
    for i,j,k, i_list, j_list, k_list in (callback):
        add_res += sum([Children[c].happiness(i) for c in i_list]) - sum([Children[c].happiness(i) for c in Gifts_list[i]])
        add_res += sum([Children[c].happiness(j) for c in j_list]) - sum([Children[c].happiness(j) for c in Gifts_list[j]])
        add_res += sum([Children[c].happiness(k) for c in k_list]) - sum([Children[c].happiness(k) for c in Gifts_list[k]])
        
        Gifts_list[i] = i_list
        Gifts_list[j] = j_list
        Gifts_list[k] = k_list
        
#    print("elapsed:{:.2f} add_res:{}".format(time.time()-st_time, add_res))
    
    if (t+1) % 10 == 0:
        d = time.time()-st_time
        print("t = {} : increased = {} elapsed = {:.2f}".format(t, add_res, d))
        if d>timelimit:
            break


## In[18]:
#add_res = 0
#st_time = time.time()
#
#for t in range(200000):
#    
#    i, j, k = np.random.choice(1000, 3, replace=False)
#    i_list, j_list, k_list = swap3_greedy_all(i, j, k)
#    
#    add_res += sum([Children[c].happiness(i) for c in i_list]) - sum([Children[c].happiness(i) for c in Gifts_list[i]])
#    add_res += sum([Children[c].happiness(j) for c in j_list]) - sum([Children[c].happiness(j) for c in Gifts_list[j]])
#    add_res += sum([Children[c].happiness(k) for c in k_list]) - sum([Children[c].happiness(k) for c in Gifts_list[k]])
#    
#    Gifts_list[i] = i_list
#    Gifts_list[j] = j_list
#    Gifts_list[k] = k_list
#        
#    
#    
#    if (t+1) % 1000 == 0:
#        print("elapsed:{:.2f} add_res:{}".format(time.time()-st_time, add_res))
#        print("t = {} : increased {}".format(t, add_res))
#
## In[18]:
#
#for t in range(200000, 1000000):
#    
#    
#    i, j, k = np.random.choice(1000, 3, replace=False)
#    i_list, j_list, k_list = swap3_greedy_all(i, j, k)
#    
#    add_res += sum([Children[c].happiness(i) for c in i_list]) - sum([Children[c].happiness(i) for c in Gifts_list[i]])
#    add_res += sum([Children[c].happiness(j) for c in j_list]) - sum([Children[c].happiness(j) for c in Gifts_list[j]])
#    add_res += sum([Children[c].happiness(k) for c in k_list]) - sum([Children[c].happiness(k) for c in Gifts_list[k]])
#    
#    Gifts_list[i] = i_list
#    Gifts_list[j] = j_list
#    Gifts_list[k] = k_list
#        
#    if (t+1) % 1000 == 0:
#        print("t = " + str(t) +" : increased " + str(add_res))


# 4000000000 : 1  
# 400000000  : 0.1  
# 40000000   : 0.01  

# In[19]:

def avg_normalized_happiness_(pred, child_pref, gift_pref):
    
    # check if number of each gift exceeds n_gift_quantity
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= n_gift_quantity
                
    # check if twins have the same gift
    # for t1 in range(0,twins,2):
    #     twin1 = pred[t1]
    #     twin2 = pred[t1+1]
    #     assert twin1[1] == twin2[1]
    
    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)
    
    for row in pred:
        child_id = row[0]
        gift_id = row[1]
        
        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0 
        assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(gift_pref[child_id]==gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = ( n_child_pref - np.where(child_pref[gift_id]==child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness
    
    # print(max_child_happiness, max_gift_happiness
    print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) ,         ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))
    return float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) + np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity)




# In[20]:

res_list_temp = [[] for i in range(1000000)]
for j in range(1000):
    for i in Gifts_list[j]:
        res_list_temp[i] = [i, j]


# In[21]:

#avg_normalized_happiness(res_list_temp, child_pref, gift_pref)


# In[22]:

avg_normalized_happiness_(res_list_temp, child_pref, gift_pref)


# In[23]:

Gifts_list_2 = Gifts_list.copy()


# In[24]:

# modify for twins
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



# In[26]:

#avg_normalized_happiness(res_list_temp, child_pref, gift_pref)


# In[27]:


df = pd.DataFrame(res_list_temp,
                  columns=['ChildId','GiftId'])
df.to_csv(outfile, index=False, compression='gzip')






