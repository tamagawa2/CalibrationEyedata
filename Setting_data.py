import numpy as np
from sklearn.neighbors import KernelDensity
import DataSetting_Cali
import random
import json
import os

    

def weight_setting(weight, file_label):
    
    
    unique = np.unique(file_label)
    re = np.ones((weight.shape[0]))
    
    # for i in range(unique.shape[0]):
        
    #     label_count = 1.0 / np.count_nonzero(file_label == unique[i])
    #     if unique[i] == 2:
    #         label_count = label_count * 1
        
    #     weight_kari = weight[file_label == unique[i]]
    #     weight_kari = weight_kari.reshape(-1, 1)
        
    #     weight_index = np.argwhere(file_label == unique[i]).reshape(-1)
    #     re_kari = np.zeros((weight_kari.shape[0]))
        
        
    #     # kde = KernelDensity(kernel='gaussian', bandwidth=100).fit(weight_kari)
    #     # re_kari = kde.score_samples(weight_kari)
    #     # re_kari = np.exp(re_kari)
    
    #     for j in range(re_kari.shape[0]):
    #         kariti_x = np.abs(weight_kari[j, 0])
    #         alpha = 1 / 10.0
    #         kariti_y = (1.0 / (np.power(alpha * kariti_x, 1.0) + 1.0))
    #         # re_kari[j] =  (kariti_y / re_kari[j]) * label_count
            
    #         if kariti_x < 100:
    #             kariti_y = 1.0
    #         else:
    #             kariti_y = 0.0
    #         re_kari[j] = kariti_y
    #         re[weight_index[j]] = re_kari[j]
        
            
    return re.reshape(-1, 1)

def split_gakusyuu_data(data):
    train_data = DataSetting_Cali.Eye_Data()
    valid_data = DataSetting_Cali.Eye_Data()
    test_data = DataSetting_Cali.Eye_Data()
    
    range_ = 50
    max_range = 1000
    i_range = int(max_range / range_)
    for i in range(-i_range, i_range):
        
        jyoukenn1 = (data.send_weight[:, 0] > i * range_) & (data.send_weight[:, 0] <= ((i + 1) * range_))
        jyoukenn2 = (data.send_file_label[:] > i * range_) & (data.send_file_label[:] <= ((i + 1) * range_))
        data_kari = DataSetting_Cali.Eye_Data()
        data_kari = split_data(data, jyoukenn1, jyoukenn2)
        
        data_kari_array = []
        count1 = 0
        count2 = 0
        for j in range(data_kari.send_file_number.shape[0]):
            s1 = count1
            e1 = int(count1 + data_kari.send_file_number[j])
            s2 = count2
            e2 = s2 + 1
            data_kari_kari = DataSetting_Cali.Eye_Data()
            
            data_kari_kari = slice_split_data(data_kari, s1, e1, s2, e2)
            data_kari_array.append(data_kari_kari)
            count1 = e1
            count2 = e2
            
            
            
        random.shuffle(data_kari_array)
        a_size = len(data_kari_array)
        for j in range(a_size):
            if j < int(a_size * 0.6):
                train_data = ketugou_Eye_Data(train_data, data_kari_array[j])
            elif j >= int(a_size * 0.6) and j < int(a_size * 0.8):
                valid_data = ketugou_Eye_Data(valid_data, data_kari_array[j])
            else:
                test_data = ketugou_Eye_Data(test_data, data_kari_array[j])
                
    
    return train_data, valid_data, test_data



def zurasi_gakusyuu_data(data, zurasi):
    re = DataSetting_Cali.Eye_Data()
    
    count1 = 0
    count2 = 0
    for i in range(data.send_file_number.shape[0]):
        
        s1 = count1
        e1 = int(count1 + data.send_file_number[i])
        s2 = count2
        e2 = s2 + 1
        data_kari = slice_split_data(data, s1, e1, s2, e2)
        
        
        zurasi_kari = int(zurasi[i])
        target = data_kari.send_x_array - data_kari.send_y_array
        target = np.roll(target, zurasi_kari, 0)
        data_kari.send_y_array = data_kari.send_x_array - target
        data_kari.send_file_number = data_kari.send_file_number - zurasi_kari
        data_kari.send_file_zurasi = np.array([zurasi_kari])
        data_kari.send_file_jyunnbann = np.zeros_like(data_kari.send_file_jyunnbann.shape)
        data_kari = slice_split_data(data_kari, zurasi_kari, e1 - s1, 0, 1)
        re = ketugou_Eye_Data(re, data_kari)
        
        count1 = e1
        count2 = e2
        
    return re


def zurasi_get_file(data, dirn, filen):
    path = dirn + filen
    file = open(path + ".bin", 'rb')
    b = file.read()
    s = b.decode()
    e = json.loads(s)
    file.close()
    
    zurasi_class = save_zurasi()
    zurasi_class.zurasi = e["zurasi"]
    
    return zurasi_gakusyuu_data(data, zurasi_class.zurasi)
    
    
    
def zurasi_set_file(zurasi_class, dirn, filen):
    path = dirn + filen
    file = open(path + ".bin", 'rb')
    jsondata = json.dumps(zurasi_class, cls=MyEncoder)
    file = open(path + ".bin", "wb")
    file.write(jsondata.encode())
    file.close()
    
    
            
def slice_split_data(data, s1, e1, s2, e2):
    
    re = DataSetting_Cali.Eye_Data()
    re.send_x_array = data.send_x_array[s1:e1]
    re.send_y_array = data.send_y_array[s1:e1]
    re.send_v_array = data.send_v_array[s1:e1]
    re.send_v_y_array = data.send_v_y_array[s1:e1]
    re.send_time_array = data.send_time_array[s1:e1]
    re.send_weight = data.send_weight[s1:e1]
    re.send_label = data.send_label[s1:e1]
    re.send_file_mode = data.send_file_mode[s1:e1]
    re.send_file_jyunnbann = data.send_file_jyunnbann[s1:e1]
    
    re.send_file_label = data.send_file_label[s2:e2]
    re.send_file_number = data.send_file_number[s2:e2]
    re.send_file_zurasi = data.send_file_zurasi[s2:e2]
    
    re.file_label_count = re.send_file_number.shape[0]
    
    return re

def split_data(data, jyoukenn1, jyoukenn2):
    
    re = DataSetting_Cali.Eye_Data()
    re.send_x_array = data.send_x_array[jyoukenn1]
    re.send_y_array = data.send_y_array[jyoukenn1]
    re.send_v_array = data.send_v_array[jyoukenn1]
    re.send_v_y_array = data.send_v_y_array[jyoukenn1]
    re.send_time_array = data.send_time_array[jyoukenn1]
    re.send_weight = data.send_weight[jyoukenn1]
    re.send_label = data.send_label[jyoukenn1]
    re.send_file_mode = data.send_file_mode[jyoukenn1]
    re.send_file_jyunnbann = data.send_file_jyunnbann[jyoukenn1]
    
    re.send_file_label = data.send_file_label[jyoukenn2]
    re.send_file_number = data.send_file_number[jyoukenn2]
    re.send_file_zurasi = data.send_file_zurasi[jyoukenn2]
    
    re.file_label_count = re.send_file_number.shape[0]
    
    return re
    
        

def ketugou_Eye_Data(ketugou_data, data):
    re = DataSetting_Cali.Eye_Data()
    if ketugou_data.send_x_array.shape[0] == 0:
        re.send_x_array = data.send_x_array.copy()
        re.send_y_array = data.send_y_array.copy()
        re.send_v_array = data.send_v_array.copy()
        re.send_v_y_array = data.send_v_y_array.copy()
        re.send_time_array = data.send_time_array.copy()
        re.send_weight = data.send_weight.copy()
        re.send_label = data.send_label.copy()
        re.send_file_mode = data.send_file_mode.copy()
        re.send_file_jyunnbann = data.send_file_jyunnbann.copy()
        
        re.send_file_label = data.send_file_label.copy()
        re.send_file_number = data.send_file_number.copy()
        re.send_file_zurasi = data.send_file_zurasi.copy()
        
        re.file_label_count = data.file_label_count
    else:
        re.send_x_array = np.vstack([ketugou_data.send_x_array, data.send_x_array])
        re.send_y_array = np.vstack([ketugou_data.send_y_array, data.send_y_array])
        re.send_v_array = np.vstack([ketugou_data.send_v_array, data.send_v_array])
        re.send_v_y_array = np.vstack([ketugou_data.send_v_y_array, data.send_v_y_array])
        re.send_time_array = np.vstack([ketugou_data.send_time_array, data.send_time_array])
        re.send_weight = np.vstack([ketugou_data.send_weight, data.send_weight])
        re.send_label = np.vstack([ketugou_data.send_label, data.send_label])
        re.send_file_mode = np.hstack([ketugou_data.send_file_mode, data.send_file_mode])
        re.send_file_jyunnbann = np.hstack([ketugou_data.send_file_jyunnbann, ketugou_data.send_file_jyunnbann[-1] + 1 + data.send_file_jyunnbann])
        
        re.send_file_label = np.hstack([ketugou_data.send_file_label, data.send_file_label])
        re.send_file_number = np.hstack([ketugou_data.send_file_number, data.send_file_number])
        re.send_file_zurasi = np.hstack([ketugou_data.send_file_zurasi, data.send_file_zurasi])
        
        re.file_label_count = ketugou_data.file_label_count + data.file_label_count
    
    return re
    
class save_zurasi:
    zurasi = []
    
class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, save_zurasi):
            return {"zurasi": o.zurasi}
        
        return json.JSONEncoder.default(self, o)
    

def mean_loss(loss, setting_label):
    re = np.zeros_like(loss)
    count = 0
    for i in range(loss.shape[0]):
        
        count = count + 1
        if i != loss.shape[0] - 1:
            if setting_label[i] != setting_label[i + 1]:
                re[i - count + 1 :i + 1] = np.mean(loss[i - count + 1:i + 1])
                count = 0
                
        else:
            re[i - count + 1 :i + 1] = np.mean(loss[i - count:i + 1])
                
        
    return re
    