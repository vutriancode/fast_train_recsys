from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import torch
import numpy as np

from CONFIG import *

class Training:

    def __init__(self, model):
        self.model = model

    def train(self, train_loader, optimizer,epoch, best_rmse, best_mae, device="cpu"):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            batch_nodes_u, batch_nodes_v, labels_list = data
            optimizer.zero_grad()
            loss = self.model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            if i % 5000 == 0:
                print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                    epoch, i, running_loss / 1, best_rmse, best_mae))
                running_loss = 0.0
        return 0
    #print(train_u)
    def test(self,test_loader, device):
        self.model.eval()
        tmp_pred = []
        target = []
        with torch.no_grad():
            for test_u, test_v, tmp_target in test_loader:

                test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
                val_output = self.model.forward(test_u, test_v)
                tmp_pred.append(list(val_output.data.cpu().numpy()))
                target.append(list(tmp_target.data.cpu().numpy()))
        tmp_pred = np.array(sum(tmp_pred, []))
        target = np.array(sum(target, []))
        expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
        mae = mean_absolute_error(tmp_pred, target)
        return expected_rmse, mae