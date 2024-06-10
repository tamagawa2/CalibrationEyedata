import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io
import PyTorch_mlp_cali_y_kannsuu as torch_mlp
import numpy as np
import DataSetting_Cali
import DataSetting

class net_class (nn.Module):
    def __init__(self):
        super(net_class, self).__init__()
        
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class CaliCoodinate:
    def __init__(self) -> None:
        self.eyedata = DataSetting_Cali.Eye_Data()
        self.train_loss_list = []
        self.train_acc_list = []
        self.valid_loss_list = []
        self.valid_acc_list = []
        self.num_epochs = 0
        self.outputpath = ""
        self.net = net_class()
        self.x1 = ""
        self.y1 = ""
            
    
    def DoCaliCoodinate(self, input_path, outputpath, batch_size, num_epochs):
        
        self.outputpath = outputpath
        self.num_epochs = num_epochs
        
        M = 1
        file_Cali = "/Data"
        data = DataSetting_Cali.re_Cali_velo(input_path, file_Cali, 1, M, 3)
        x1 = np.hstack([data.send_x_array[:, 0:1, 0].copy(), data.send_x_array[:, 1:2, 0].copy()])
        y1 = np.hstack([data.send_y_array[:, 0:1, 0].copy(), data.send_y_array[:, 1:2, 0].copy()])
        self.x1 = x1
        self.y1 = y1
        xx1= torch.from_numpy(x1.copy()).float()
        yy1 = torch.from_numpy(y1.copy()).float()
        Dataset = torch.utils.data.TensorDataset(xx1, yy1) 
        train_size = int(len(Dataset) * 0.8) 
        test_size = int(len(Dataset) * 0.01) 
        valid_size = int(len(Dataset)) - train_size - test_size

        train, test, valid = torch.utils.data.random_split(Dataset, [train_size, test_size, valid_size])

        train_l = torch.utils.data.DataLoader(dataset = train, batch_size = batch_size, shuffle = True) 
        test_l = torch.utils.data.DataLoader(dataset = test, batch_size = batch_size, shuffle = True) 
        valid_l = torch.utils.data.DataLoader(dataset = valid, batch_size = batch_size, shuffle = True)
        
        self.net = net_class()
        criterion_1 = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.001)
        
        consoletext = ""
        self.train_loss_list = []
        self.train_acc_list = []
        self.valid_loss_list = []
        self.valid_acc_list = []
        
        for epoch in range(num_epochs):
            t_loss, t_acc = torch_mlp.train_fn(self.net, train_l, criterion_1, optimizer)
            v_loss, v_acc = torch_mlp.valid_fn(self.net, valid_l, criterion_1, optimizer)
            
            self.train_loss_list.append(t_loss)
            self.train_acc_list.append(t_acc)
            self.valid_loss_list.append(v_loss)
            self.valid_acc_list.append(v_acc)

            if epoch % (num_epochs / 10) == 0:
                consoletext += f"epoch: {epoch} t_loss: {t_loss} v_loss{v_loss}\n"
                yield consoletext
        
    def Output_DoCaliCoodinste(self):
        
        fig1 = plt.figure()
        plt.plot(range(self.num_epochs), self.train_loss_list, c = "r", label = "train_loss")
        plt.plot(range(self.num_epochs), self.valid_loss_list, c = "b", label = "valid_loss")
        plt.legend()
        
        
        
        # # Matplotlibのグラフを画像として取得
        # buffer = io.BytesIO()
        # plt.savefig(buffer, format='png')
        # buffer.seek(0)
        # image1 = Image.open(buffer)
        # plt.close()  # グラフのクリア
        
        
        
        fig2 = plt.figure()
        y_ = self.net(torch.from_numpy(self.x1).float()).detach().numpy()
        x = self.x1[:, 0]
        y = self.y1[:, 0]
        plt.scatter(x, y, s = 1, c = "b")

        x = self.x1[:, 0]
        y = y_[:, 0]
        plt.scatter(x, y, s = 1, c = "r")
        
        
        # Matplotlibのグラフを画像として取得
        # buffer = io.BytesIO()
        # plt.savefig(buffer, format='png')
        # buffer.seek(0)
        # image2 = Image.open(buffer)
        # plt.close()  # グラフのクリア
        
        
        output_mlp = DataSetting.output_mlp()
        mlp_data = DataSetting.mlp_data()
        mlp_data.coef = self.net.fc1.weight.data.numpy().T.tolist()
        mlp_data.intercept = self.net.fc1.bias.data.numpy().tolist()
        output_mlp.hidden_layer.append(mlp_data)
        mlp_data = DataSetting.mlp_data()
        mlp_data.coef = self.net.fc2.weight.data.numpy().T.tolist()
        mlp_data.intercept = self.net.fc2.bias.data.numpy().tolist()
        output_mlp.hidden_layer.append(mlp_data)
        mlp_data = DataSetting.mlp_data()
        mlp_data.coef = self.net.fc3.weight.data.numpy().T.tolist()
        mlp_data.intercept = self.net.fc3.bias.data.numpy().tolist()
        output_mlp.hidden_layer.append(mlp_data)
        output_mlp.of_from = 1
        output_mlp.of_to = 1
        dirPre = self.outputpath
        filePre = "/py_cali_co"
        
        name = dirPre + filePre
        DataSetting.Pre_output(output_mlp, name)
            
        return fig1, fig2
        
        
        
