import pickle
import pandas as pd
import os
import csv
import sys
from post_embedding import *
from CONFIG import *
"""
with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
content_files = open(os.path.join(LINK_DATA,"data_preparation.csv"),"r")
data=csv.reader(content_files)

print(data)
print(len(item_dict))
content_dict=dict()
a=0
for i in data:
    try:
        if i[1]!="" or i[1]!=" ":
            content_dict[i[0]]=i[1]
            a = a+1
    except:
        print(i[0])
print(a)
#print(content_dict)
# with open(os.path.join(LINK_DATA,"contens.picke"),"wb") as out_put_file:
    pickle.dump(content_dict,out_put_file)"""
with open(os.path.join(LINK_DATA,"content.picke"),"rb") as out_put_file:
    content=pickle.load(out_put_file)
content_e=dict()
sys.argv[1]=int(sys.argv[1])
a=0
if sys.argv[1]<4:
    print(sys.argv[1])
    for i in list(content.keys())[sys.argv[1]*20000:(sys.argv[1]+1)*20000]:
        a+=1
        print(a)
        content_e[i] = embedding_document(content[i]).detach().numpy()
    with open(os.path.join(LINK_DATA,"content_e{}.pickle".format(sys.argv[1])),"wb") as out_put_file:
        pickle.dump(content_e,out_put_file)
else:
    print(sys.argv[1])
    for i in list(content.keys())[sys.argv[1]*20000:]:
        a+=1
        print(a)
        content_e[i] = embedding_document(content[i]).detach().numpy()
    with open(os.path.join(LINK_DATA,"content_e{}.pickle".format(sys.argv[1])),"wb") as out_put_file:
        pickle.dump(content_e,out_put_file)


    