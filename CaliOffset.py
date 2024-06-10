import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io
import PyTorch_mlp_cali_x_pre_mtl_kannsuu as torch_mlp
import numpy as np
import DataSetting_Cali
import DataSetting
import Setting_data


class net_class (nn.Module):
    
    def __init__(self, limit_y_t):
        super(net_class, self).__init__()
        self.limit_y_t = limit_y_t
        #relu
        self.relu = nn.ReLU()
        #encoder
        self.conv1 = nn.Conv1d(1, 16, kernel_size= 5, padding=2, stride = 2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2, stride = 2)
        #bottleneck layer
        self.conv4 = nn.Conv1d(32, 32, kernel_size=5, padding=2, stride = 1)
        self.conv5 = nn.Conv1d(32, 32, kernel_size=5, padding=2, stride = 1)
        #decoder
        self.de_conv1 = nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2, stride = 2)
        self.de_conv2 = nn.ConvTranspose1d(16, 1, kernel_size=7, padding=2, stride = 2)
        
    def forward(self, x):
        #encoder
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        #bottleneck layer
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        #decoder
        x = self.de_conv1(x)
        x = self.relu(x)
        x = self.de_conv2(x)
        
        x = x.reshape(x.shape[0], -1)
        return x[:, :self.limit_y_t]



class CaliOffset:
    def __init__(self) -> None:
        self.eyedata = DataSetting_Cali.Eye_Data()
        self.train_loss_1_list = []
        self.train_loss_1_array_list = []
        self.valid_loss_1_list = []
        self.valid_loss_1_array_list = []
        self.M = 15
        self.num_epochs = 0
        self.outputpath = ""
        self.net = net_class(self.M)
        self.x1 = ""
        self.y1 = ""
        
    def DoCaliOffset(self, 
                     caliof_S_path, caliof_T_path, caliof_F_path, output_path,
                     batch_size, num_epochs, M,
                     ):
        
        self.outputpath = output_path
        self.num_epochs = num_epochs
        self.M = M
        file_Cali = "/Data"
      
        dir_C = caliof_S_path
        data_S = DataSetting_Cali.re_Cali_velo(dir_C, file_Cali, 0, M, 0)
        dir_C = caliof_T_path
        data_T = DataSetting_Cali.re_Cali_velo(dir_C, file_Cali, 0, M, 1)
        dir_C = caliof_F_path
        data_F = DataSetting_Cali.re_Cali_velo(dir_C, file_Cali, 0, M, 2)
        
        data = DataSetting_Cali.Eye_Data()
        data = Setting_data.ketugou_Eye_Data(data, data_F)
        data = Setting_data.ketugou_Eye_Data(data, data_T)
        data = Setting_data.ketugou_Eye_Data(data, data_S)
        train, valid, test = Setting_data.split_gakusyuu_data(data)
        
        x_t = train.send_v_array[:, 0, :]
        y_t = train.send_y_array[:, 0, :]
        w_l_t = train.send_weight[:, 0].reshape(-1, 1)
        w_t = Setting_data.weight_setting(train.send_weight[:, 0], train.send_file_mode)
        l_t = train.send_label[:, 0:1]
        z_t = train.send_x_array[:, 0, :]
        time_t = train.send_time_array

        x_v = valid.send_v_array[:, 0, :]
        y_v = valid.send_y_array[:, 0, :]
        w_l_v = valid.send_weight[:, 0].reshape(-1, 1)
        w_v = Setting_data.weight_setting(valid.send_weight[:, 0], valid.send_file_mode)
        l_v = valid.send_label[:, 0:1]
        z_v = valid.send_x_array[:, 0, :]
        time_v = valid.send_time_array

        x_test = test.send_v_array[:, 0, :]
        y_test = test.send_y_array[:, 0, :]
        w_l_test = test.send_weight[:, 0].reshape(-1, 1)
        w_test = test.send_weight[:]
        l_test = test.send_label[:, 0:1]
        z_test = test.send_x_array[:, 0, :]
        time_test = test.send_time_array

        t_y_t = self.M
        limit_y_t = self.M
        
        torch_x_t = torch.from_numpy(x_t).float().reshape(x_t.shape[0], 1, x_t.shape[1])
        torch_y_t = torch.from_numpy(y_t).float()[:, :limit_y_t]
        torch_w_l_t = torch.from_numpy(w_l_t).float()
        torch_w_t = torch.from_numpy(w_t).float()
        torch_l_t = torch.from_numpy(l_t).long()

        torch_x_v = torch.from_numpy(x_v).float().reshape(x_v.shape[0], 1, x_v.shape[1])
        torch_y_v = torch.from_numpy(y_v).float()[:, :limit_y_t]
        torch_w_l_v = torch.from_numpy(w_l_v).float()
        torch_w_v = torch.from_numpy(w_v).float()
        torch_l_v = torch.from_numpy(l_v).long()

        torch_x_test = torch.from_numpy(x_test).float().reshape(x_test.shape[0], 1, x_test.shape[1])
        torch_y_test = torch.from_numpy(y_test).float()[:, :limit_y_t]
        torch_w_l_test = torch.from_numpy(w_l_test).float()
        torch_w_test = torch.from_numpy(w_test).float()
        torch_l_test = torch.from_numpy(l_test).long()
        
        Dataset_t = torch.utils.data.TensorDataset(torch_x_t, torch_y_t, torch_w_l_t, torch_w_t, torch_l_t)
        Dataset_v = torch.utils.data.TensorDataset(torch_x_v, torch_y_v, torch_w_l_v, torch_w_v, torch_l_v)
        Dataset_test = torch.utils.data.TensorDataset(torch_x_test, torch_y_test, torch_w_l_test, torch_w_test, torch_l_test)

        train_l = torch.utils.data.DataLoader(dataset = Dataset_t, batch_size = batch_size, shuffle = True)
        valid_l = torch.utils.data.DataLoader(dataset = Dataset_v, batch_size = batch_size, shuffle = True)
        test_l = torch.utils.data.DataLoader(dataset = Dataset_test, batch_size = batch_size, shuffle = True)

        y_unique = np.unique(l_t)
        self.net = net_class(M)
        criterion_1 = torch_mlp.L1Loss_weight_jyoukenn(c = 5.0, overshoot_limi = 20)
        optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=0.001)
        
        consoletext = ""
        self.train_loss_1_list = []
        self.train_loss_1_array_list = []
        self.valid_loss_1_list = []
        self.valid_loss_1_array_list = []
        
        
        for epoch in range(num_epochs):

            t_loss, t_loss_array = torch_mlp.train_fn(self.net, train_l, criterion_1, optimizer)
            v_loss, v_loss_array = torch_mlp.valid_fn(self.net, valid_l, criterion_1, optimizer)
            
            self.train_loss_1_list.append(t_loss)
            self.valid_loss_1_list.append(v_loss)
            print(epoch)
            
            if epoch == 0:
                for i in range(len(t_loss_array)):
                    self.train_loss_1_array_list.append([])
                    self.train_loss_1_array_list[i].append(t_loss_array[i])
                for i in range(len(v_loss_array)):
                    self.valid_loss_1_array_list.append([])
                    self.valid_loss_1_array_list[i].append(v_loss_array[i])
                    
            else:
                
                for i in range(len(t_loss_array)):
                    self.train_loss_1_array_list[i].append(t_loss_array[i])
                for i in range(len(v_loss_array)):
                    self.valid_loss_1_array_list[i].append(v_loss_array[i])
                

            if epoch % (num_epochs / 10) == 0:
                consoletext += f"epoch: {epoch} t_loss: {t_loss} v_loss{v_loss}\n"
                yield consoletext
        
    def Output_DoCaliOffset(self):
        
        
        fig1 = plt.figure()
        plt.plot(range(self.num_epochs), self.train_loss_1_list, c = "r", label = "t_loss")
        plt.plot(range(self.num_epochs), self.valid_loss_1_list, c = "b", label = "v_loss")
        plt.legend()
        
        
        cols = 1
        rows = (len(self.train_loss_1_array_list))
        fig2, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten() 
        for i in range(len(self.train_loss_1_array_list)):
            axes[i].plot(range(self.num_epochs), self.train_loss_1_array_list[i], c = "r", label = "t_loss")
            axes[i].plot(range(self.num_epochs), self.valid_loss_1_array_list[i], c = "b", label = "v_loss")
            axes[i].legend()
            axes[i].set_title(f"重み{i * 100} {i * 100}ピクセル先のデータを見ていたときは")
        
        plt.tight_layout()

        output_conv1d = DataSetting.output_Conv1d()

        conv = DataSetting.Conv1d()
        conv.weight = self.net.conv1.weight.detach().numpy().tolist()
        conv.bias = self.net.conv1.bias.detach().numpy().tolist()
        conv.padding = self.net.conv1.padding[0]
        conv.stride = self.net.conv1.stride[0]
        output_conv1d.Conv1d_array.append(conv)

        conv = DataSetting.Conv1d()
        conv.weight = self.net.conv2.weight.detach().numpy().tolist()
        conv.bias = self.net.conv2.bias.detach().numpy().tolist()
        conv.padding = self.net.conv2.padding[0]
        conv.stride = self.net.conv2.stride[0]
        output_conv1d.Conv1d_array.append(conv)

        # conv = DataSetting.Conv1d()
        # conv.weight = net.conv3.weight.detach().numpy().tolist()
        # conv.bias = net.conv3.bias.detach().numpy().tolist()
        # conv.padding = net.conv3.padding[0]
        # conv.stride = net.conv3.stride[0]
        # output_conv1d.Conv1d_array.append(conv)

        conv = DataSetting.Conv1d()
        conv.weight = self.net.conv4.weight.detach().numpy().tolist()
        conv.bias = self.net.conv4.bias.detach().numpy().tolist()
        conv.padding = self.net.conv4.padding[0]
        conv.stride = self.net.conv4.stride[0]
        output_conv1d.Conv1d_array.append(conv)


        conv = DataSetting.Conv1d()
        conv.weight = self.net.conv5.weight.detach().numpy().tolist()
        conv.bias = self.net.conv5.bias.detach().numpy().tolist()
        conv.padding = self.net.conv5.padding[0]
        conv.stride = self.net.conv5.stride[0]
        output_conv1d.Conv1d_array.append(conv)


        conv = DataSetting.Conv1d()
        conv.weight = self.net.de_conv1.weight.detach().numpy().tolist()
        conv.bias = self.net.de_conv1.bias.detach().numpy().tolist()
        conv.padding = self.net.de_conv1.padding[0]
        conv.stride = self.net.de_conv1.stride[0]
        output_conv1d.Conv1d_array.append(conv)

        conv = DataSetting.Conv1d()
        conv.weight = self.net.de_conv2.weight.detach().numpy().tolist()
        conv.bias = self.net.de_conv2.bias.detach().numpy().tolist()
        conv.padding = self.net.de_conv2.padding[0]
        conv.stride = self.net.de_conv2.stride[0]
        output_conv1d.Conv1d_array.append(conv)

        # conv = DataSetting.Conv1d()
        # conv.weight = net.de_conv3.weight.detach().numpy().tolist()
        # conv.bias = net.de_conv3.bias.detach().numpy().tolist()
        # conv.padding = net.de_conv3.padding[0]
        # conv.stride = net.de_conv3.stride[0]
        # output_conv1d.Conv1d_array.append(conv)


        dirPre = self.outputpath
        filePre = "/pytorch_cali_x_pre_All"


        name = dirPre + filePre
        DataSetting.Pre_output(output_conv1d, name)
        
            
        return fig1, fig2, 

    


