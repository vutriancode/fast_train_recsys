import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
import random
from Attention import Attention

class PostEncode(nn.Module):

    def __init__(self, u2e,r2e, p2e, embed_dim, contents_embed_dim, pu_history, pr_history, device="cpu"):
        super(PostEncode,self).__init__()
        self.u2e = u2e
        self.r2e = r2e
        self.p2e = p2e
        self.device = device 
        #self.contents_embedding = contents_embedding
        self.w_e = nn.Linear(contents_embed_dim, embed_dim).to(device)
        self.embed_dim = embed_dim
        self.w_1 = nn.Linear(2*embed_dim, embed_dim).to(device)
        self.w_2 = nn.Linear(embed_dim,embed_dim).to(device)
        self.attention =Attention(embed_dim,device=device)
        self.o_w = nn.Linear(2 * self.embed_dim, self.embed_dim).to(device)
        self.pu_history = pu_history 
        self.pr_history = pr_history
        #self.pr_content = pr_content


    def forward(self, nodes):

        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        u_embed = []
        r_embed = []
        post_rep= []
        for index, i in enumerate(nodes):
            i=int(i.numpy())
            iii = random.randint(0,len(self.pu_history[i])-1)

            u_embed.append(self.pu_history[i][iii])
            r_embed.append(self.pr_history[i][iii])
            post_rep.append(self.p2e[i])
            
            #with torch.no_grad():
                #post_rep = self.contents_embedding(self.pr_content[i])
        
        post_rep = torch.cuda.FloatTensor(post_rep)
        post_rep = F.relu(self.w_e(post_rep))
        u_embed, r_embed = self.u2e.weight[u_embed], self.r2e.weight[r_embed]
        x = torch.cat((u_embed,r_embed),1)
        x = F.relu(self.w_1(x))
        o = F.relu(self.w_2(x))
        att_history = torch.cat(( post_rep,o),1)
        att_history = F.relu(self.o_w(att_history))

        return att_history
