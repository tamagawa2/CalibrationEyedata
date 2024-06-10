
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def train_fn(model, train_loader, criterion_1, optimizer, device='cpu'):
    
    # 1epoch training 
    train_loss = 0.0
    train_acc = 0.0
    
    num_train = 0

    # model 学習モードに設定
    model.train()

    for i, (x_1, y_1) in enumerate(train_loader):
        
        # batch数を累積
        num_train += len(y_1)
        
        # 勾配をリセット
        optimizer.zero_grad()
        # 推論
        outputs_1= model(x_1)
        # lossを計算
        loss_1 = criterion_1(outputs_1, y_1)
        
        
        loss = loss_1
        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()
        
        # lossを累積
        kari_loss = loss.item()
#         print("kari_loss", kari_loss / len(offset))
        train_loss += kari_loss
        
        #　なんかaccuracy
        
        acc = 0
        
        
        train_acc += acc # train_acc に結果を蓄積
        
    
    train_loss = train_loss / num_train
    train_acc = train_acc / num_train
    
    return train_loss, train_acc



def valid_fn(model, valid_loader, criterion_1, optimizer, device='cpu'):
    
    # 評価用のコード
    valid_loss = 0.0
    val_acc = 0.0
    
    num_valid = 0

    # model 評価モードに設定
    model.eval()

    # 評価の際に勾配を計算しないようにする
    with torch.no_grad():
        for i, (x_1, y_1) in enumerate(valid_loader):
            num_valid += len(y_1)
            
            outputs_1= model(x_1)
            
            loss_1 = criterion_1(outputs_1, y_1)
            
            
                
            loss = loss_1
            
            valid_loss += loss.item()
            
            acc = 0
            
                
            val_acc += acc
            
        valid_loss = valid_loss / num_valid
        val_acc = val_acc / num_valid
        
    return valid_loss, val_acc



def run(model, train_loader, valid_loader, criterion_1, optimizer, num_epochs, epoch_print):

    device='cpu'
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    

    for epoch in range(num_epochs):

        t_loss, t_acc = train_fn(model, train_loader, criterion_1, optimizer, device=device)
        v_loss, v_acc = valid_fn(model, valid_loader, criterion_1, optimizer, device=device)

        if epoch % epoch_print == 0:
            print("epoch:", epoch, 
                  "t_loss:", t_loss,
                  "v_loss:", v_loss
                  )
            
        train_loss_list.append(t_loss)
        train_acc_list.append(t_acc)
        valid_loss_list.append(v_loss)
        valid_acc_list.append(v_acc)

    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list