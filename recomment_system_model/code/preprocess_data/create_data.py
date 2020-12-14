import pandas as pd
import numpy as np
import pickle
import os
from CONFIG import *

with open(os.path.join(LINK_DATA,"training.pickle"),"rb") as out_put_file:
    a = pickle.load(out_put_file)
user_post =dict()
post_user =dict()
user_rate =dict()
post_rate = dict()
print(list(a.values()).count(0))
print(list(a.values()).count(1))
print(list(a.values()).count(2))
print(list(a.values()).count(3))
print(list(a.values()).count(4))
print(list(a.values()).count(5))

for i in list(a.keys()):
    i=list(i)
    #print(i)
    if i[0] not in user_post.keys():
        user_post[i[0]] = [i[1]]
        user_rate[i[0]] = [a[tuple(i)]]
    else:
        user_post[i[0]].append(i[1])
        user_rate[i[0]].append(a[tuple(i)])
    if i[1] not in post_user.keys():
        post_user[i[1]] = [i[0]]
        post_rate[i[1]] = [a[tuple(i)]]
    else:
        post_user[i[1]].append(i[0])
        post_rate[i[1]].append(a[tuple(i)])
with open(os.path.join(LINK_DATA,"data2.pickle"),"rb") as out_put_file:
    user_post,user_rate,post_user,post_rate = pickle.load(out_put_file)
mmm =[]
for j in [[i,len(post_user[i])] for i in post_user.keys()]:
    if j[1]>10:
        mmm.append(j[0])
print(len(mmm))
training= dict()
aa=0
for i in a.keys():
    aa=aa+1
    print(aa)
    i=list(i)
    if i[1] in mmm:
        training[tuple(i)]=a[tuple(i)]
with open(os.path.join(LINK_DATA,"training2.pickle"),"wb") as out_put_file:
    pickle.dump(training,out_put_file)
print(max([len(i) for i in user_post.values()]))
print(min([len(i) for i in user_post.values()]))
print(max([len(i) for i in post_user.values()]))
print(min([len(i) for i in post_user.values()]))


     