import numpy as np
import json
import os
import Filter


def GetEyeCali(dirn, filen, flag):
    
    file = open(dirn + filen + ".bin", 'rb')
    b = file.read()
    s = b.decode()
    e = json.loads(s)
    
    l = len(e["CalibrationTarget"])

    x_s_1 = 2
    y_s_1 = 2
    
    data_x = np.zeros((l, x_s_1))
    data_y = np.zeros((l, y_s_1))
    
    data_avg_y = np.zeros((l, y_s_1))
    
    bunnkatu = e["bunnkatu"]
    kari_x = np.zeros((bunnkatu, x_s_1))
    kari_y = np.zeros((bunnkatu, y_s_1))
    
   
   
    for i in range(l):
        
        P_X = e["CalibrationPointArray"][i]["X"]
        P_Y = e["CalibrationPointArray"][i]["Y"]
        
        E_X = e["CalibrationEyeArray"][i]["X"]
        E_Y = e["CalibrationEyeArray"][i]["Y"]
        
        kari_x[i % bunnkatu, 0] = E_X
        kari_x[i % bunnkatu, 1] = E_Y
        
        kari_y[i % bunnkatu, 0] = P_X
        kari_y[i % bunnkatu, 1] = P_Y
        
        
        
        if (i + 1) % bunnkatu == 0:
            
            avg_x = np.zeros((x_s_1))
            
                
            for j in range(bunnkatu):
                for k in range(avg_x.shape[0]):
                    avg_x[k] = avg_x[k] + kari_x[j, k]
                
            for k in range(avg_x.shape[0]):
                avg_x[k] = avg_x[k] / bunnkatu
                
           
            for j in range(bunnkatu):
                for k in range(avg_x.shape[0]):
                    
                    if flag == 0:
                        data_x[i - bunnkatu + 1 + j, k] = kari_x[j, k]
                    else:
                        if k < 2:
                            data_x[i - bunnkatu + 1 + j, k] = avg_x[k]
                        else:
                            data_x[i - bunnkatu + 1 + j, k] = avg_x[k]
                
                for k in range(y_s_1):   
                    data_y[i - bunnkatu + 1 + j, k] = kari_y[j, k]
                
        
    return data_x, data_y, 


def re_CaliData(dirn, filen, flag):
    
    path = dirn
    
    count_file = 0
    #ディレクトリの中身分ループ
    for file_name in os.listdir(path):
        file_path = os.path.join(path,file_name)
        if os.path.isfile(file_path) and file_name != ".DS_Store":
            count_file = count_file + 1
    nn = count_file
    
    send_x = np.zeros((0, 2))
    send_y = np.zeros((0, 2))
    
    
    for i in range(nn):
        data_x, data_y = GetEyeCali(dirn, filen + str(i), flag)
        
        send_x = np.vstack([send_x, data_x])
        send_y = np.vstack([send_y, data_y])
            

    return send_x, send_y
    
    
    
def bunnri_data(dirn, filen, f_f_flag):
    
    path = dirn
    
    count_file = 0
    for file_name in os.listdir(path):
        file_path = os.path.join(path,file_name)
        if os.path.isfile(file_path) and file_name != ".DS_Store":
            count_file = count_file + 1
    nn = count_file
    
    send_timestumps = np.zeros((0))
    send_time_array = np.zeros((0), dtype=np.int64)
    send_flick_flick_time_array = np.zeros((0))
    
    for i in range(nn):
        file = open(dirn + filen + str(i) + ".bin", 'rb')
        b = file.read()
        s = b.decode()
        e = json.loads(s)

        timestumps = np.zeros((len(e["CalibrationTarget"])))
        
        for j in range(timestumps.shape[0]):
            timestumps[j] = e["timestumps"][j]
            
        time_array = np.zeros((len(e["time_list"])), dtype=np.int64)
        
        for j in range(time_array.shape[0]):
            time_array[j] = int(e["time_list"][j])
            
        
        if f_f_flag == 0:
            flick_flick_time_array = np.zeros((1))
        else:
            flick_flick_time_array = np.zeros((len(e["flick_flick_time_list"])))
            for j in range(flick_flick_time_array.shape[0]):
                flick_flick_time_array[j] = e["flick_flick_time_list"][j]
            
        send_timestumps = np.hstack([send_timestumps, timestumps])
        send_time_array = np.hstack([send_time_array, time_array])
        send_flick_flick_time_array = np.hstack([send_flick_flick_time_array, flick_flick_time_array])
        
        
            
        
        
    return send_timestumps, send_time_array, send_flick_flick_time_array


class Eye_Data:
    def __init__(self):
        self.send_x_array = np.zeros((0, 0, 0))
        self.send_y_array = np.zeros((0, 0, 0))
        self.send_v_array = np.zeros((0, 0, 0))
        self.send_v_y_array = np.zeros((0, 0, 0))
        self.send_time_array = np.zeros((0, 0))
        self.send_weight = np.zeros((0, 0))
        self.send_label = np.zeros((0, 0))
        self.send_file_mode = np.zeros((0))
        self.send_file_jyunnbann = np.zeros((0))
        
        self.send_file_label = np.zeros((0))
        self.send_file_number = np.zeros((0))
        self.send_file_zurasi = np.zeros((0))
        
        self.file_label_count = 0


def re_Cali_velo(dirn, filen, flag, M, set_f):
    
    data_x, data_y = re_CaliData(dirn, filen, flag)
    timestumps, time_array, flick_flick_time_array = bunnri_data(dirn, filen, 0)
    
    time_array_count = 0
    
    send_x_array = np.zeros((0, data_x.shape[1], M))
    
    send_y_array = np.zeros((0, data_y.shape[1], M))
    
    send_v_array = np.zeros((0, data_x.shape[1], M))
    
    send_v_y_array = np.zeros((0, data_y.shape[1], M))
    
    send_time_array = np.zeros((0, M))
    
    send_weight = np.zeros((0, M))
    
    send_label = np.zeros((0, M))
    send_file_mode = np.zeros((0)) 
    send_file_jyunnbann = np.zeros((0)) 
    
    send_file_label = np.zeros((0))
    send_file_number = np.zeros((0))
    
    file_label_count = 0
    
    
    for i in range(time_array.shape[0]):
        
        x_array_kari = np.zeros((time_array[i] - 1, data_x.shape[1]))
        x_y_array_kari = np.zeros((time_array[i] - 1, data_y.shape[1]))
        v_array_kari = np.zeros((time_array[i] - 1, data_x.shape[1]))
        v_y_array_kari = np.zeros((time_array[i] - 1, data_y.shape[1]))
        time_array_kari = np.zeros((time_array[i] - 1))
        weight_array_kari = np.zeros((time_array[i] - 1))
        label_array_kari = np.zeros((time_array[i] - 1))
        file_mode_array_kari = np.zeros((time_array[i] - 1))
        file_jyunnbann_array_kari = np.zeros((time_array[i] - 1))
        
        file_label_array_kari = 0
        
        if time_array[i] >= M + 1:
            
        
            for j in range(time_array[i]):

                if j != 0:
                    
                    dt = (timestumps[time_array_count + j] - timestumps[time_array_count + j - 1]) / 1000000.0
                    
                    time_array_kari[j - 1] = (timestumps[time_array_count + j] - timestumps[time_array_count]) / 1000000.0

                    for k in range(v_array_kari.shape[1]):
                        x_array_kari[j - 1, k] = data_x[time_array_count + j, k]
                        v_array_kari[j - 1, k] = ( data_x[time_array_count + j, k] - data_x[time_array_count + j - 1, k] ) / dt
                        
                    for k in range(v_y_array_kari.shape[1]):
                        x_y_array_kari[j - 1, k] = data_y[time_array_count + j, k]
                        v_y_array_kari[j - 1, k] = ( data_y[time_array_count + j, k] - data_y[time_array_count + j - 1, k] ) / dt
                        
                        
                   
            
                        
            target_kari = x_array_kari[:, :2] - x_y_array_kari

            
            if set_f == 0:
                x_y_array_kari = np.zeros_like(x_y_array_kari)
                weight_array_kari = np.zeros_like(weight_array_kari)
                label_array_kari = np.full(label_array_kari.shape, 2)
                file_mode_array_kari = np.full(file_mode_array_kari.shape, set_f)
                file_label_array_kari = 0
                file_jyunnbann_array_kari = np.full(file_jyunnbann_array_kari.shape, file_label_count)
                file_label_count = file_label_count + 1
                
                
            elif set_f == 1:
                
                for b in range(1): #真面目につくるんだったらweight_array_kariらへんから変えないと...
                    for a in range(x_y_array_kari.shape[0]):
                        if time_array_kari[a] >= 3.5 and time_array_kari[a] <= 4.1:
                            kariti_x = (target_kari[a, b] - target_kari[a - 1, b]) / (time_array_kari[a] - time_array_kari[a - 1])
                            label_y = int(np.abs(kariti_x) / 100)
                
                
                
                weight_array_kari = np.full(weight_array_kari.shape, kariti_x)
                label_array_kari = np.full(label_array_kari.shape, label_y)
                file_mode_array_kari = np.full(file_mode_array_kari.shape, set_f)
                file_jyunnbann_array_kari = np.full(file_jyunnbann_array_kari.shape, file_label_count)
                file_label_array_kari = kariti_x
                file_label_count = file_label_count + 1
                
                
                for b in range(1):
                    for a in range(x_y_array_kari.shape[0]):
                        if time_array_kari[a] < 3.0:
                            x_y_array_kari[a, b] = 0
                
                
                    
                lim_time = 3.0
                save_lim = 0
                
                lim_out_lim = 5.0
                save_out_lim = 0
                
                for a in range(time_array_kari.shape[0]):
                    if time_array_kari[a] < lim_time:
                        save_lim = a - M
                        
                    if time_array_kari[a] <= lim_out_lim and time_array_kari[a] >= lim_time:
                        save_out_lim = a
                        
                
                x_array_kari = x_array_kari[save_lim:save_out_lim]
                x_y_array_kari = x_y_array_kari[save_lim:save_out_lim]
                v_array_kari = v_array_kari[save_lim:save_out_lim]               
                v_y_array_kari = v_y_array_kari[save_lim:save_out_lim]
                time_array_kari = time_array_kari[save_lim:save_out_lim]
                weight_array_kari = weight_array_kari[save_lim:save_out_lim]
                label_array_kari = label_array_kari[save_lim:save_out_lim]
                file_mode_array_kari = file_mode_array_kari[save_lim:save_out_lim]
                file_jyunnbann_array_kari = file_jyunnbann_array_kari[save_lim:save_out_lim]
                            
                

            elif set_f == 2:
                
                for b in range(1):
                    for a in range(x_y_array_kari.shape[0]):
                        if time_array_kari[a] >= 3.75 and time_array_kari[a] <= 4.0:
                            kariti_x = target_kari[a, b] - 960.0
                            label_y = 0
                            kariti_y = kariti_x
                            
                            label_y = int(np.abs(kariti_x) / 100)
                
                weight_array_kari = np.full(weight_array_kari.shape, kariti_y)
                label_array_kari = np.full(label_array_kari.shape, label_y)
                file_mode_array_kari = np.full(file_mode_array_kari.shape, set_f)
                file_jyunnbann_array_kari = np.full(file_jyunnbann_array_kari.shape, file_label_count)
                file_label_array_kari = kariti_y
                file_label_count = file_label_count + 1
                
                for b in range(1):
                    for a in range(x_y_array_kari.shape[0]):
                        if time_array_kari[a] < 3.0:
                            x_y_array_kari[a, b] = 0
                  
                    
                lim_time = 3.0
                save_lim = 0
                
                lim_out_lim = 4.25
                save_out_lim = 0
                
                for a in range(time_array_kari.shape[0]):
                    if time_array_kari[a] < lim_time:
                        save_lim = a - M
                        
                    if time_array_kari[a] <= lim_out_lim and time_array_kari[a] >= lim_time:
                        save_out_lim = a

                x_array_kari = x_array_kari[save_lim:save_out_lim]
                x_y_array_kari = x_y_array_kari[save_lim:save_out_lim]
                v_array_kari = v_array_kari[save_lim:save_out_lim]               
                v_y_array_kari = v_y_array_kari[save_lim:save_out_lim]
                time_array_kari = time_array_kari[save_lim:save_out_lim]
                weight_array_kari = weight_array_kari[save_lim:save_out_lim]
                label_array_kari = label_array_kari[save_lim:save_out_lim]
                file_mode_array_kari = file_mode_array_kari[save_lim:save_out_lim]
                file_jyunnbann_array_kari = file_jyunnbann_array_kari[save_lim:save_out_lim]
                            

            if x_array_kari.shape[0] >= M:
                
                send_time_array_kari = np.zeros((time_array_kari.shape[0] - (M - 1), M))
                for a in range(send_time_array_kari.shape[0]):
                    for c in range(send_time_array_kari.shape[1]):
                        send_time_array_kari[a, c] = time_array_kari[a + M - 1 - c]

                v_array = np.zeros((v_array_kari.shape[0] - (M - 1), v_array_kari.shape[1], M))
                for b in range(v_array.shape[1]):
                    for a in range(v_array.shape[0]):
                        for c in range(M):
                            v_array[a, b, c] = v_array_kari[a + M - 1 - c, b]
                            
                v_y_array = np.zeros((v_y_array_kari.shape[0] - (M - 1), v_y_array_kari.shape[1], M))
                for b in range(v_y_array.shape[1]):
                    for a in range(v_y_array.shape[0]):
                        for c in range(M):
                            v_y_array[a, b, c] = v_y_array_kari[a + M - 1 - c, b]
                
                x_array = np.zeros((v_array.shape[0], data_x.shape[1], M))
                for b in range(x_array.shape[1]):
                    for a in range(x_array.shape[0]):
                        for c in range(M):
                            x_array[a, b, c] = x_array_kari[a + M - 1 - c, b]
                            
                            
                y_array = np.zeros((v_y_array.shape[0], data_y.shape[1], M))
                for b in range(y_array.shape[1]):
                    for a in range(y_array.shape[0]):
                        for c in range(M):
                            y_array[a, b, c] = x_y_array_kari[a + M - 1 - c, b]
                            
                weight_array = np.zeros((weight_array_kari.shape[0] - (M - 1), M))
                for a in range(weight_array.shape[0]):
                    for c in range(M):
                        weight_array[a, c] = weight_array_kari[a + M - 1 - c]
                        
                label_array = np.zeros((label_array_kari.shape[0] - (M - 1), M))
                for a in range(label_array.shape[0]):
                    for c in range(M):
                        label_array[a, c] = label_array_kari[a + M - 1 - c]
                        
                
                file_mode_array = np.zeros((file_mode_array_kari.shape[0] - (M - 1)))
                for a in range(file_mode_array.shape[0]):
                    for c in range(M):
                        file_mode_array[a] = file_mode_array_kari[a + M - 1 - c]
                
                file_jyunnbann_array = np.zeros((file_jyunnbann_array_kari.shape[0] - (M - 1)))
                for a in range(file_jyunnbann_array.shape[0]):
                    for c in range(M):
                        file_jyunnbann_array[a] = file_jyunnbann_array_kari[a + M - 1 - c]
                    
                
                file_label_array = np.array([file_label_array_kari])
                
                
                send_x_array = np.vstack([send_x_array, x_array])
                send_y_array = np.vstack([send_y_array, y_array])
                send_v_array = np.vstack([send_v_array, v_array]) 
                send_v_y_array = np.vstack([send_v_y_array, v_y_array]) 
                send_time_array = np.vstack([send_time_array, send_time_array_kari])
                send_weight = np.vstack([send_weight, weight_array]) 
                send_label = np.vstack([send_label, label_array]) 
                
                send_file_mode = np.hstack([send_file_mode, file_mode_array])
                send_file_jyunnbann = np.hstack([send_file_jyunnbann, file_jyunnbann_array])
                
                send_file_label = np.hstack([send_file_label, file_label_array])
                send_file_number = np.hstack([send_file_number, np.array([x_array.shape[0]])])
                
            else:
                file_label_count = file_label_count - 1
                
            
        time_array_count = time_array_count + time_array[i]
        
    eye_data = Eye_Data()  
    eye_data.send_x_array = send_x_array
    eye_data.send_y_array = send_y_array
    eye_data.send_v_array = send_v_array
    eye_data.send_v_y_array = send_v_y_array
    eye_data.send_time_array = send_time_array
    eye_data.send_weight = send_weight
    eye_data.send_label = send_label
    eye_data.send_file_mode = send_file_mode
    eye_data.send_file_jyunnbann = send_file_jyunnbann
    eye_data.send_file_label = send_file_label
    eye_data.send_file_number = send_file_number
    eye_data.send_file_zurasi = np.zeros_like(eye_data.send_file_number)
    eye_data.file_label_count = file_label_count
    

    return eye_data


