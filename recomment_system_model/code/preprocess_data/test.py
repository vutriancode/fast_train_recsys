import pickle
import os 
from CONFIG import *
with open(os.path.join(LINK_DATA,"training.pickle"),"rb") as out_put_file:
    traning=pickle.load(out_put_file)
train_u = []
train_v = []
train_r = []
for i in traning.keys():
    train_u.append(i[0])
    train_v.append(i[1])
    train_r.append(traning[i])
print(len(train_r))