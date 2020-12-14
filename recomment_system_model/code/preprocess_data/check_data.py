import pickle
from CONFIG import *
import os

with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
#with open(os.path.join(LINK_DATA,"contens.picke"),"rb") as out_put_file:
#    content=pickle.load(out_put_file)
print(ir_dict)
print(min([len(ir) for ir in ir_dict.values()]))
"""a=dict()
for i in content.keys():
    try:
        #print(i)
        a[item_dict[i]]= content[i]
    except:
        pass
print(len(a))
print(a)
with open(os.path.join(LINK_DATA,"content.picke"),"wb") as out_put_file:
    pickle.dump(a,out_put_file)"""