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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import numpy as np

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
train_u = []
train_v = []
train_r = []
for i in traning.keys():
    train_u.append(i[0])
    train_v.append(i[1])
    train_r.append(traning[i])
#print(train_u)
def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:

            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
#testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
#                                             torch.FloatTensor(test_r))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
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
#test(score,"cpu",train_loader)
m.train(train_loader,optimizer,20,999,999,device=device)