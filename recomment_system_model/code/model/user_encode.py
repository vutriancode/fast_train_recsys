import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
from Attention import Attention

class UserEncode(nn.Module):

    def __init__(self, u2e, r2e, i2e, embed_dim, up_history, ur_history, device="cpu"):
        super(UserEncode,self).__init__()
        self.u2e = u2e
        self.r2e = r2e
        self.i2e = i2e
        self.device = device 
        #self.contents_embedding = contents_embedding
        self.w_e = nn.Linear(768, embed_dim).to(device)
        self.embed_dim = embed_dim
        self.w_1 = nn.Linear(2*embed_dim, embed_dim).to(device)
        self.w_2 = nn.Linear(embed_dim,embed_dim).to(device)
        self.w_3 = nn.Linear(embed_dim*2,embed_dim).to(device)
        self.up_history = up_history
        self.ur_history = ur_history
        self.attention =Attention(embed_dim,device)
    def forward(self, nodes):

        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        p_embed=[]
        r_embed=[]
        for index, i in enumerate(nodes):

            i=int(i.numpy())
            #print(len(self.up_history[i]))
            #print(len(self.ur_history[i]))
            iii = random.randint(0,len(self.ur_history[i])-1)
            #print(iii)
            p_embed.append(self.up_history[i][iii])
            r_embed.append(self.ur_history[i][iii])
        p_embed=torch.cuda.FloatTensor(p_embed, device=self.device)
        u_rep = self.u2e.weight[nodes.numpy()].to(self.device)
        #p_embed = self.i2e.weight[j].to(self.device)
        p_embed = F.relu(self.w_e(p_embed))
        r_embed = self.r2e.weight[r_embed].to(self.device)
        #r_embed = self.post_embedding()
        
        r_embed=r_embed.reshape(-1,self.embed_dim)
        x = torch.cat((p_embed,r_embed),1)
        x = F.relu(self.w_1(x))
        x = F.relu(self.w_2(x)).to(self.device)
        x = torch.cat((u_rep,x),1)
        o = F.relu(self.w_3(x))
        return o


