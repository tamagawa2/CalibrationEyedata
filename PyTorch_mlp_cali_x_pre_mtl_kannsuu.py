
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class L1Loss_weight_jyoukenn(nn.Module):
    def __init__(self, c, overshoot_limi): 
        super(L1Loss_weight_jyoukenn, self).__init__()
        self.c = c
        self.overshoot_limi = overshoot_limi
        
    def forward(self, outputs, y1, weight_label, weight, label):

        x = torch.abs(y1 - outputs) 
        x_overshoot = torch.where((torch.abs(y1) - torch.abs(outputs)) > 0, x, x * self.c)
        x = torch.where(np.abs(weight_label) > self.overshoot_limi, x_overshoot, x)
        x = x * weight
        loss = torch.mean(x)
        loss_array = []
        label_unique = torch.unique(label)
        
        for i in range(label_unique.shape[0]):
            losskari = torch.where(label == label_unique[i], x, x * 0)
            losskari = torch.mean(losskari)
            loss_array.append(losskari)
            
        
        return loss, loss_array
    

def train_fn(model, train_loader, criterion_1, optimizer, device='cpu'):
    
    loss_stock = 0.0
    loss_stock_array = []
    num = 0
    # model 学習モードに設定
    model.train()

    for i, (x_1, y_1, weight_label, weight, label) in enumerate(train_loader):
        
        num += len(y_1)
        optimizer.zero_grad()
        outputs_1= model(x_1)
        loss_1, loss_array_1 = criterion_1(outputs_1, y_1, weight_label, weight, label)
        loss = loss_1
        loss.backward()
        optimizer.step()

        loss_stock += loss.item()
        if i == 0:
            for j in range(len(loss_array_1)):
                loss_stock_array.append(loss_array_1[j].item())
                
        else:
            for j in range(len(loss_array_1)):
                loss_stock_array[j] += loss_array_1[j].item()
            
    loss_stock = loss_stock / num
    for i in range(len(loss_stock_array)):
        loss_stock_array[i] = loss_stock_array[i] / num
        
        
        
    return loss_stock, loss_stock_array



def valid_fn(model, valid_loader, criterion_1, optimizer, device='cpu'):
    
    loss_stock = 0.0
    loss_stock_array = []
    num = 0

    # model 評価モードに設定
    model.eval()

    # 評価の際に勾配を計算しないようにする
    with torch.no_grad():
        for i, (x_1, y_1, weight_label, weight, label) in enumerate(valid_loader):
        
            num += len(y_1)
            outputs_1= model(x_1)
            loss_1, loss_array_1 = criterion_1(outputs_1, y_1, weight_label, weight, label)
            
            
            loss = loss_1
            loss_stock += loss.item()
            if i == 0:
                for j in range(len(loss_array_1)):
                    loss_stock_array.append(loss_array_1[j].item())
                    
            else:
                for j in range(len(loss_array_1)):
                    loss_stock_array[j] += loss_array_1[j].item()
            
        loss_stock = loss_stock / num
        for i in range(len(loss_stock_array)):
            loss_stock_array[i] = loss_stock_array[i] / num
        
        
        
    return loss_stock, loss_stock_array



def run(model, train_loader, valid_loader, criterion_1, optimizer, num_epochs, epoch_print):

    device='cpu'
    train_loss_1_list = []
    train_loss_1_array_list = []
    valid_loss_1_list = []
    valid_loss_1_array_list = []
    
    

    for epoch in range(num_epochs):

        t_loss, t_loss_array = train_fn(model, train_loader, criterion_1, optimizer, device=device)
        v_loss, v_loss_array = valid_fn(model, valid_loader, criterion_1, optimizer, device=device)
        
        train_loss_1_list.append(t_loss)
        valid_loss_1_list.append(v_loss)
        
        
        if epoch == 0:
            for i in range(len(t_loss_array)):
                train_loss_1_array_list.append([])
                train_loss_1_array_list[i].append(t_loss_array[i])
            for i in range(len(v_loss_array)):
                valid_loss_1_array_list.append([])
                valid_loss_1_array_list[i].append(v_loss_array[i])
                
        else:
            for i in range(len(t_loss_array)):
                train_loss_1_array_list[i].append(t_loss_array[i])
            for i in range(len(v_loss_array)):
                valid_loss_1_array_list[i].append(v_loss_array[i])
            

        if epoch % epoch_print == 0:
            
            print("epoch:", epoch, 
                "t_loss:", t_loss,
                "v_loss:", v_loss,
                )
        
        

    return train_loss_1_list, train_loss_1_array_list, valid_loss_1_list, valid_loss_1_array_list


