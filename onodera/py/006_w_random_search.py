# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""

nohup python -u 006_w_random_search.py > log1.txt &
nohup python -u 006_w_random_search.py > log2.txt &
nohup python -u 006_w_random_search.py > log3.txt &
nohup python -u 006_w_random_search.py > log4.txt &

"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from collections import Counter
from ortools.graph import pywrapgraph
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

n_children = 1000000 # n children to give
n_gift_type = 1000 # n types of gifts available
n_gift_quantity = 1000 # each type of gifts are limited to this quantity
n_gift_pref = 100 # number of gifts a child ranks
n_child_pref = 1000 # number of children a gift ranks
twins = math.ceil(0.04 * n_children / 2.) * 2    # 4% of all population, rounded to the closest number
triplets = math.ceil(0.005 * n_children / 3.) * 3    # 0.5% of all population, rounded to the closest number
ratio_gift_happiness = 2
ratio_child_happiness = 2

seed = np.random.randint(9999)
print('seed:',seed)

gift_pref = pd.read_csv('../input/child_wishlist_v2.csv.zip',header=None).drop(0, 1).values
child_pref = pd.read_csv('../input/gift_goodkids_v2.csv.zip',header=None).drop(0, 1).values


def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    # in case of large numbers, using floor division
    return a * b // math.gcd(a, b)

def avg_normalized_happiness(pred, child_pref, gift_pref):
    
    # check if number of each gift exceeds n_gift_quantity
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= n_gift_quantity
                
    # check if triplets have the same gift
    for t1 in np.arange(0,triplets,3):
        triplet1 = pred[t1]
        triplet2 = pred[t1+1]
        triplet3 = pred[t1+2]
        # print(t1, triplet1, triplet2, triplet3)
        assert triplet1[1] == triplet1[1] and triplet2[1] == triplet3[1]
                
    # check if twins have the same gift
    for t1 in np.arange(triplets,triplets+twins,2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        # print(t1)
        assert twin1[1] == twin2[1]

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
    
    print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) , \
        ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))

    # to avoid float rounding error
    # find common denominator
    # NOTE: I used this code to experiment different parameters, so it was necessary to get the multiplier
    # Note: You should hard-code the multipler to speed up, now that the parameters are finalized
    denominator1 = n_children*max_child_happiness
    denominator2 = n_gift_quantity*max_gift_happiness*n_gift_type
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    # # usually denom1 > demon2
    return float(math.pow(total_child_happiness*multiplier,3) + math.pow(np.sum(total_gift_happiness),3)) / float(math.pow(common_denom,3))
    # return math.pow(float(total_child_happiness)/(float(n_children)*float(max_child_happiness)),2) + math.pow(np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity),2)

#random_sub = pd.read_csv('../input/sample_submission_random_v2.csv').values.tolist()
#print(avg_normalized_happiness(random_sub, child_pref, gift_pref))

#gift_pref.shape, child_pref.shape



class Child(object):
    
    def __init__(self, idx, prefer):
        
        self.idx = idx
        self.prefer_dict = dict()
        
        for i in range(prefer.shape[0]):
            self.prefer_dict[prefer[i]] = [12*(prefer.shape[0] - i), -6]
    
    
    def add_gifts_prefer(self, giftid, score):
        
        if giftid in self.prefer_dict.keys():
            self.prefer_dict[giftid][1] = 6*score
        else:
            self.prefer_dict[giftid] = [-6, 6*score]
        
        return None
        
    
    def happiness(self, giftid):
        
        return self.prefer_dict.get(giftid, [-6, -6])

class Child_twin(object):
    
    def __init__(self, idx, prefer1, prefer2):
        
        self.idx = idx
        self.prefer_dict = dict()
        
        for p in list(set(list(prefer1) + list(prefer2))):
            score = 0
            if p in list(prefer1):
                score += 2*(100 - list(prefer1).index(p))
            else:
                score -= 1
            if p in list(prefer2):
                score += 2*(100 - list(prefer2).index(p))
            else:
                score -= 1
            self.prefer_dict[p] = [3*score, -6]
    
    
    def add_gifts_prefer(self, giftid, score):
        
        if giftid in self.prefer_dict.keys():
            self.prefer_dict[giftid][1] = 3*score
        else:
            self.prefer_dict[giftid] = [-6, 3*score]
        
        return None
        
    
    def happiness(self, giftid):
        
        return self.prefer_dict.get(giftid, [-6, -6])

class Child_triplet(object):
    
    def __init__(self, idx, prefer1, prefer2, prefer3):
        
        self.idx = idx
        self.prefer_dict = dict()
        
        for p in list(set(list(prefer1) + list(prefer2) + list(prefer3))):
            score = 0
            if p in list(prefer1):
                score += 2*(100 - list(prefer1).index(p))
            else:
                score -= 1
            if p in list(prefer2):
                score += 2*(100 - list(prefer2).index(p))
            else:
                score -= 1
            if p in list(prefer3):
                score += 2*(100 - list(prefer3).index(p))
            else:
                score -= 1
            self.prefer_dict[p] = [2*score, -6]
    
    
    def add_gifts_prefer(self, giftid, score):
        
        if giftid in self.prefer_dict.keys():
            self.prefer_dict[giftid][1] = 2*score
        else:
            self.prefer_dict[giftid] = [-6, 2*score]
        
        return None
        
    
    def happiness(self, giftid):
        
        return self.prefer_dict.get(giftid, [-6, -6])

Children = []
for i in range(0, 5001, 3):
    Children.append(Child_triplet(i, gift_pref[i], gift_pref[i+1], gift_pref[i+2]))
    Children.append(Child_triplet(i+1, gift_pref[i], gift_pref[i+1], gift_pref[i+2]))
    Children.append(Child_triplet(i+2, gift_pref[i], gift_pref[i+1], gift_pref[i+2]))
for i in range(5001, 45001, 2):
    Children.append(Child_twin(i, gift_pref[i], gift_pref[i+1]))
    Children.append(Child_twin(i+1, gift_pref[i], gift_pref[i+1]))
Children = Children + [Child(i, gift_pref[i]) for i in range(45001, 1000000)]

for j in range(1000):
    cf = child_pref[j]
    done_list = []
    for i in range(cf.shape[0]):
        if cf[i] <= 5000 and cf[i] not in done_list:
            if cf[i] % 3 == 0:
                cid1 = cf[i]
                cid2 = cf[i] + 1
                cid3 = cf[i] + 2
                done_list.append(cid2)
                done_list.append(cid3)
            elif cf[i] % 3 == 1:
                cid1 = cf[i] - 1
                cid2 = cf[i]
                cid3 = cf[i] + 1
                done_list.append(cid1)
                done_list.append(cid3)
            else:
                cid1 = cf[i] - 2
                cid2 = cf[i] - 1
                cid3 = cf[i]
                done_list.append(cid1)
                done_list.append(cid2)
            if cid1 in list(cf):
                score_ = 2*(cf.shape[0] - list(cf).index(cid1))
            else:
                score_ = -1
            if cid2 in list(cf):
                score_ += 2*(cf.shape[0] - list(cf).index(cid2))
            else:
                score_ += -1
            if cid3 in list(cf):
                score_ += 2*(cf.shape[0] - list(cf).index(cid3))
            else:
                score_ += -1
            Children[cid1].add_gifts_prefer(j, score_)
            Children[cid2].add_gifts_prefer(j, score_)
            Children[cid3].add_gifts_prefer(j, score_)
        elif cf[i] <= 45000 and cf[i] not in done_list:
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
        elif cf[i] > 45000:
            Children[cf[i]].add_gifts_prefer(j, 2*(cf.shape[0] - i))


print("W_CHILD, W_GIFTS, score")

while True:
    
    W_CHILD = np.random.randint(8000, 99999999)
    W_GIFTS = np.random.randint(1, 5)
    
    
    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []
    
    # triplets
    for i in range(0, 5001, 3):
        for g in Children[i].prefer_dict.keys():
            start_nodes.append(1000000+g)
            end_nodes.append(i)
            capacities.append(3)
            unit_costs.append(-W_CHILD*(Children[i].prefer_dict[g][0] + 6)-W_GIFTS*(Children[i].prefer_dict[g][1] + 6))
            
    # triplets
    for i in range(5001, 45001, 2):
        for g in Children[i].prefer_dict.keys():
            start_nodes.append(1000000+g)
            end_nodes.append(i)
            capacities.append(2)
            unit_costs.append(-W_CHILD*(Children[i].prefer_dict[g][0] + 6)-W_GIFTS*(Children[i].prefer_dict[g][1] + 6))
            
    # other children
    for i in range(45001, 1000000):
        
        for g in Children[i].prefer_dict.keys():
            start_nodes.append(1000000+g)
            end_nodes.append(i)
            capacities.append(1)
            unit_costs.append(-W_CHILD*(Children[i].prefer_dict[g][0] + 6)-W_GIFTS*(Children[i].prefer_dict[g][1] + 6))
    
    min_cost_flow_1 = pywrapgraph.SimpleMinCostFlow()
    
    # add Arc
    # gift -> children
    for i in range(len(start_nodes)):
        min_cost_flow_1.AddArcWithCapacityAndUnitCost(
            int(start_nodes[i]), int(end_nodes[i]), int(capacities[i]), int(unit_costs[i])
        )
        
    # children -> 1001000 : collection
    for i in range(0, 5001, 3):
        min_cost_flow_1.AddArcWithCapacityAndUnitCost(
            int(i), int(1001000), int(3), int(0)
        )
    for i in range(5001, 45001, 2):
        min_cost_flow_1.AddArcWithCapacityAndUnitCost(
            int(i), int(1001000), int(2), int(0)
        )
    for i in range(45001, 1000000):
        min_cost_flow_1.AddArcWithCapacityAndUnitCost(
            int(i), int(1001000), int(1), int(0)
        )
        
    # gift -> 1001001 : dust_gift
    for i in range(1000):
        min_cost_flow_1.AddArcWithCapacityAndUnitCost(
            int(1000000+i), int(1001001), int(1000), int(0)
        )
        
    # 1001001 -> 1001000 : dust_path
    min_cost_flow_1.AddArcWithCapacityAndUnitCost(
            int(1001001), int(1001000), int(1000000), int(0)
        )
    
    
    # add Supply
    for i in range(1000):
        min_cost_flow_1.SetNodeSupply(int(1000000+i), int(1000))
    
    # children
    for i in range(0, 5001, 3):
        min_cost_flow_1.SetNodeSupply(int(i), int(0))
    for i in range(5001, 45001, 2):
        min_cost_flow_1.SetNodeSupply(int(i), int(0))
    for i in range(45001, 1000000):
        min_cost_flow_1.SetNodeSupply(int(i), int(0))
    
    min_cost_flow_1.SetNodeSupply(int(1001001), int(0)) 
    min_cost_flow_1.SetNodeSupply(int(1001000), int(-1000000)) 
    
    min_cost_flow_1.Solve()
    
    assignment = [-1]*1000000
    twins_differ = []
    triplets_differ = []
    
    for i in range(min_cost_flow_1.NumArcs()):
        if min_cost_flow_1.Flow(i) != 0 and min_cost_flow_1.Head(i) < 1000000:
            c = min_cost_flow_1.Head(i)
            g = min_cost_flow_1.Tail(i)
            f = min_cost_flow_1.Flow(i)
    
            if c >= 45001:
                assignment[c] = g - 1000000
    
            elif c >= 5001:
                if f == 1:
                    if assignment[c] == -1:
                        assignment[c] = g - 1000000
                        twins_differ.append([c, c+1])
                    else:
                        assignment[c+1] = g - 1000000
                elif f == 2:
                    assignment[c] = g - 1000000
                    assignment[c+1] = g - 1000000
            else:
                if f == 1:
                    if assignment[c] == -1:
                        assignment[c] = g - 1000000
                        triplets_differ.append([c, c+1, c+2])
                    elif assignment[c+1] == -1:
                        assignment[c+1] = g - 1000000
                    else:
                        assignment[c+2] = g - 1000000
                elif f == 2:
                    if assignment[c] == -1:
                        assignment[c] = g - 1000000
                        assignment[c+1] = g - 1000000
                        triplets_differ.append([c, c+1, c+2])
                    else:
                        assignment[c+1] = g - 1000000
                        assignment[c+2] = g - 1000000
                elif f == 3:
                    assignment[c] = g - 1000000
                    assignment[c+1] = g - 1000000
                    assignment[c+2] = g - 1000000
    
    CHILD_HAPPINESS = sum([Children[i].happiness(assignment[i])[0] for i in range(1000000)])*10
    SANTA_HAPPINESS = sum([Children[i].happiness(assignment[i])[1] for i in range(1000000)])
    
    OBJ = CHILD_HAPPINESS**3 + SANTA_HAPPINESS**3
    
    score = OBJ / (12000000000**3)
    print('{}, {}, {}'.format(W_CHILD, W_GIFTS, score))
    
    # wata: 0.9362730938
    
