import numpy as np
import json
import os


class EyeData :
    
    EyeDataX = []
    TargetdataX = []
    TimeStumps = []
    time = []
    
    def __init__(self):
        
        self.EyeDataX = []
        self.TargetdataX = []
        self.TimeStumps = []
        self.time = []
        
        
class EyeEnter :
    
    def __init__(self):
        
        self.v = []
        self.offset = []
        self.eye = []
        self.target = []
        self.time = []

def GetEyeData (dirn, filen) :
    eyedats = []
    
   
    path = dirn
    nn = sum(os.path.isfile(os.path.join(path,name)) for name in os.listdir(path))
    print(nn)
    
    
    for num in range(nn) :

        file = open(dirn + filen + str(num) + ".bin", 'rb')
        b = file.read()
        s = b.decode()
        e = json.loads(s)
        eyedats.append(e)

    return eyedats

def GetKetugouEyeData (dirn, filen) :
    
    #うんち！
    dates = GetEyeData(dirn, filen)
    
    eyeDatas = EyeData()
    
    for i in range(len(dates)):
        
        t = 0
        for j in range(len(dates[i]["EyeDataX"])):
            
            if j == 0 :
                eyeDatas.EyeDataX.append(dates[i]["EyeDataX"][j])
                eyeDatas.TargetdataX.append(dates[i]["TargetdataX"][j])
                eyeDatas.TimeStumps.append(dates[i]["TimeStumps"][j])
                eyeDatas.time.append(t)
                
            else :
                eyeDatas.EyeDataX.append(dates[i]["EyeDataX"][j])
                eyeDatas.TargetdataX.append(dates[i]["TargetdataX"][j])
                eyeDatas.TimeStumps.append(dates[i]["TimeStumps"][j])

                dt = (dates[i]["TimeStumps"][j] - dates[i]["TimeStumps"][j - 1]) / 1000000.0
                
                t = t + dt
                eyeDatas.time.append(t)
            #print(t)
                
    return eyeDatas

def EnterEye (data, v_n, t_n, b_time, a_time, b_n, a_n, target_zurasi, absflag):

    time_array = data["Time_Array"]
    
    v = []
    eye = []
    target = []
    time = []
    
    v_h = 0
    t_h = 0
    t = 0
    
    for i in range(len(data["EyeDataX"])):
        
        
        if i == 0:
            
            t = 0
            t_h = data["TargetdataX"][i]
            
        else :
            
            dt = (data["TimeStumps"][i] - data["TimeStumps"][i - 1]) / 1000000.0
            t = t + dt
            v_now = (data["EyeDataX"][i] - data["EyeDataX"][i - 1]) / dt
            t_h =  (1 - t_n) * t_h + t_n * data["TargetdataX"][i]
            
            if i == 1:
                
                if absflag == 0:
                    v_h = v_now
                else:
                    v_h = np.abs(v_now)
                
            else:
                
                if absflag == 0:
                    v_h = (1 - v_n) * v_h + v_n * v_now
                else:
                    v_h = (1 - v_n) * v_h + v_n * np.abs(v_now)
            
                    
                v.append(v_h)
                eye.append(data["EyeDataX"][i])
                target.append(t_h)
                time.append(t)
                
                
            
#     print("mae" + str(target))  
    target_kari = []
    for i in range(len(target)):
        if i < target_zurasi:
            target_kari.append(0)
        else:
            target_kari.append(target[i - target_zurasi])
    target = target_kari
#     print("ato" + str(target))


    send_v = []
    send_offset = []
    send_eye = []
    send_target = []
    send_time = []
    for i in range(len(time)):

        t = time[i]

        if t > time_array[b_n] * 0.001 + b_time and t < time_array[a_n] * 0.001 + a_time:

            send_v.append(v[i])
            send_offset.append(target[i] - eye[i])
            send_eye.append(eye[i])
            send_target.append(target[i])
            send_time.append(t)
            
                
    eyeenter = EyeEnter()
    eyeenter.v = send_v
    eyeenter.offset = send_offset
    eyeenter.eye = send_eye
    eyeenter.target = send_target
    eyeenter.time = send_time
    
    
    return eyeenter

    

def EnterEyeTarget (data, nn, mintime, maxtime):

    Target = []
    t = 0
    for i in range(len(data["EyeDataX"])):

        if i == 0 :

            Target.append(0)
            t = 0
        else :

            dt = (data["TimeStumps"][i] - data["TimeStumps"][i - 1]) / 1000000.0
            t = t + dt
            if t > mintime and t < maxtime :


                tt = (data["TargetdataX"][i] - data["EyeDataX"][i - 1])
                Target.append(tt)


    return Target

def ColorSetting (color0, color1, length):

    color = []

    a = color1 - color0
    t0 = 1 / length
    for i in range(length):

        k = color0 + a * (t0 * i)
        cc = (k[0], k[1], k[2])
        color.append(cc)

    return color



        
class Prediction_Data:
    
    def __init__(self):
        self.betas = [[0 for j in range(1)] for i in range(1)]
            
class svm_data:
    
    def __init__(self):
        self.kari = 0
        self.gamma = 0
        self.coef0 = 0
        self.xi = [[0 for j in range(1)] for i in range(1)]
        self.b = 0
        self.alpha = []
        self.d = 0
        
        
class mlp_data:
    
    def __init__(self):
        
        self.coef = []
        self.intercept = []
        
class output_mlp:
    
    def __init__(self):
        
        self.hidden_layer = []
        self.output_layer = []
        self.of_from = 0
        self.of_to = 0
        self.sub_layer = []
        
class Filter_keisuu:
    
    def __init__(self):
        
        self.a = []
        self.b = []
        
class Calibration_Data:
    
    def __init__(self):
        
        self.ave_head_pos = []
        
class lstm_data:
    
    def __init__(self):
        
        self.w_ii = []
        self.b_ii = []
        
        self.w_if = []
        self.b_if = []
        
        self.w_ig = []
        self.b_ig = []
        
        self.w_io = []
        self.b_io = []
        
        self.w_hi = []
        self.b_hi = []
        
        self.w_hf = []
        self.b_hf = []
        
        self.w_hg = []
        self.b_hg = []
        
        self.w_ho = []
        self.b_ho = []
        
        self.output_layer = []
        
class Conv1d:
    def __init__(self):
        self.weight = []
        self.bias = []
        self.padding = 0
        self.stride = 0
class AvgPool1d:
    def __init__(self):
        self.kernel_size = 0
class output_Conv1d:
    def __init__(self):
        self.Conv1d_array = []
        self.Avgpool_array = []
        self.fc_array = []
        
        
class MyEncoder_Pre(json.JSONEncoder):
    def default(self, o):
        

        
        if isinstance(o, Prediction_Data):
            return {"betas": o.betas}
        

        if isinstance(o, svm_data):
            return {"kari": o.kari, 
                   "gamma": o.gamma,
                   "coef0": o.coef0,
                   "xi": o.xi,
                   "b": o.b,
                   "alpha": o.alpha,
                   "d": o.d}
                     
        
        if isinstance(o, output_mlp):
            return{
                "hidden_layer": o.hidden_layer,
                "output_layer": o.output_layer,
                "of_from": o.of_from,
                "of_to": o.of_to,
                "sub_layer":o.sub_layer
            }
        
        if isinstance(o, mlp_data):
            return {
                "coef": o.coef,
                "intercept": o.intercept
            }
        
        if isinstance(o, Filter_keisuu):
            return {
                "a": o.a,
                "b": o.b
            }
        
        if isinstance(o, lstm_data):
            return {
                "w_ii": o.w_ii,
                "b_ii": o.b_ii,
                
                "w_if": o.w_if,
                "b_if": o.b_if,
                
                "w_ig": o.w_ig,
                "b_ig": o.b_ig,
                
                "w_io": o.w_io,
                "b_io": o.b_io,
                
                "w_hi": o.w_hi,
                "b_hi": o.b_hi,
                
                "w_hf": o.w_hf,
                "b_hf": o.b_hf,
                
                "w_hg": o.w_hg,
                "b_hg": o.b_hg,
                
                "w_ho": o.w_ho,
                "b_ho": o.b_ho,
                
                "output_layer": o.output_layer
                
            }
        
        if isinstance(o, Conv1d):
            return {
                "weight": o.weight,
                "bias": o.bias,
                "padding": o.padding,
                "stride": o.stride
                
            }
        
        if isinstance(o, AvgPool1d):
            return {
                "kernel_size": o.kernel_size,
            }
        
        if isinstance(o, output_Conv1d):
            return {
                "Conv1d_array": o.Conv1d_array,
                "Avgpool_array": o.Avgpool_array,
                "fc_array": o.fc_array,
            }
        
        if isinstance(o, Calibration_Data):
            return {
                "ave_head_pos": o.ave_head_pos,
            }
                  
        return json.JSONEncoder.default(self, o)
    
def Predata_to_json(data):
    
    jsondata = json.dumps(data, cls=MyEncoder_Pre)
    return jsondata

def Pre_output(data, string):
    jsondata = Predata_to_json(data)
    file = open(string + ".bin", "wb")
    file.write(jsondata.encode())
    file.close()
        
def GetPre_data(dirn, filen):
    file = open(dirn + filen + ".bin", 'rb')
    b = file.read()
    s = b.decode()
    e = json.loads(s)
    file.close()
    return e
    

def Get_Filter_Data(dirn, filen):
    
    path = dirn + filen
    file = open(dirn + filen + ".bin", 'rb')
    b = file.read()
    s = b.decode()
    e = json.loads(s)
    
    a = np.array(e["a"])
    b = np.array(e["b"])
    
    return a, b