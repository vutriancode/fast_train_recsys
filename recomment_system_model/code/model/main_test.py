#from code.model.final_model import GraphRec
import pickle
import torch 
import torch.nn as nn
import os
from load_data import *
from CONFIG import *
#from post_embedding import *
from post_encode import *
from user_encode import *
#from post_embedding import *
from final_model import *
from sklearn.model_selection import train_test_split
embed_dim = 50
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
with open(os.path.join(LINK_DATA,"data2.pickle"),"rb") as out_put_file:
    ui_dict,ur_dict,iu_dict,ir_dict  = pickle.load(out_put_file)
with open(os.path.join(LINK_DATA,"content.picke"),"rb") as out_put_file:
    content=pickle.load(out_put_file)
with open(os.path.join(LINK_DATA,"training.pickle"),"rb") as out_put_file:
    traning=pickle.load(out_put_file)
data_u = []
data_v = []
data_r = []
for i in traning.keys():
    data_u.append(i[0])
    data_v.append(i[1])
    data_r.append(traning[i])
z=1280000
train_u,train_v,train_r,test_u,test_v,test_r = \
    data_u[:z],data_v[:z],data_r[:z],data_u[z:],data_v[z:],data_r[z:]
trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                              torch.FloatTensor(test_r))
#testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
#                                             torch.FloatTensor(test_r))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

#test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
p2e = dict()
for i in range(4):
    with open(os.path.join(LINK_DATA,"content_e{}.pickle".format(i)),"rb") as out_put_file:
        p2e.update( pickle.load(out_put_file))
up_dict=dict()
for i in ui_dict.keys():
    up_dict[i] = [p2e[kk] for kk in ui_dict[i]]

print("a")
u2e = nn.Embedding(len(user_dict), embed_dim).to(device)
i2e = nn.Embedding(len(item_dict), embed_dim).to(device)
r2e = nn.Embedding(6, embed_dim).to(device)
postEncode = PostEncode(u2e, r2e,p2e, 50, 768, iu_dict,ir_dict,device=device)
userEncode = UserEncode(u2e, r2e,p2e, 50,up_dict,ur_dict,device=device)
score = GraphRec(userEncode,postEncode,r2e,device=device)
m= Training(score)
optimizer = torch.optim.RMSprop(score.parameters(), lr=0.001, alpha=0.9)
expected_rmse, mae =9999, 9999
for i in range(100):
    m.train(train_loader,optimizer,i,expected_rmse, mae,device=device)
    expected_rmse, mae = m.test(score,device,test_loader)