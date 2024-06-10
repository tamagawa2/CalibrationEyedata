from scipy import signal
import numpy as np


class Denntatsu_Kannsuu:
    
    def __init__(self, a, b):
        
        self.a = a
        self.b = b
        self.N = a.size - 1
        self.M = b.size
        
    def H_Kannsuu(self, z):
        
        A_z = 0
        for i in range(self.N + 1):
            A_z = A_z + self.a[i] * np.power(z, -i, dtype = complex)
            
        B_z = 0
        for i in range(self.M):
            B_z = B_z + self.b[i] * np.power(z, -i - 1, dtype = complex)
            
        H = A_z / (1 + B_z)
            
        return H
    
    
    
class Digital_Filter:
    
    def __init__(self, a, b, x_0):
        
        self.a = a
        self.b = b
        self.x_hozon = np.full((a.size), x_0)
        self.y_hozon = np.full((b.size), x_0)
    
    def H_Filter(self, x):
        
        self.x_hozon[1:] = self.x_hozon[:-1]
        self.x_hozon[0] = x
        
        
        y = np.dot(self.x_hozon, self.a) - np.dot(self.y_hozon, self.b)
        
        self.y_hozon[1:] = self.y_hozon[:-1]
        self.y_hozon[0] = y
        
        return y
    
    
    
class Digital_filter_matome_y_hozon:
    def __init__(self, a_array, b_array, x_0):

        self.filter_array = []
        self.y_zenntai_hozon = x_0

        for i in range(len(a_array)):
            filter_array_kari = []
            for j in range(len(a_array[i])):
                filter_array_kari.append(Digital_Filter(a_array[i][j], b_array[i][j], x_0))

            self.filter_array.append(filter_array_kari)
            
            
    def H_Filter_matome(self, x, r):
        
        y_kari = np.zeros((len(self.filter_array)))
        for i in range(len(self.filter_array)):
            y_kari[i] = x #入力
            
            self.filter_array[i][0].y_hozon[0] = self.y_zenntai_hozon  #出力
            
            for j in range(len(self.filter_array[i])):
                y_kari[i] = self.filter_array[i][j].H_Filter(y_kari[i])
            
            y_kari[i] = y_kari[i] * r[i]
            
        y = 0
        for i in range(y_kari.shape[0]):
            y = y + y_kari[i] 
        
        self.y_zenntai_hozon = y
        return y
    
    
    
    
    
    
class Digital_filter_matome:
    def __init__(self, a_array, b_array, x_0):

        self.filter_array = []

        for i in range(len(a_array)):
            filter_array_kari = []
            for j in range(len(a_array[i])):
                filter_array_kari.append(Digital_Filter(a_array[i][j], b_array[i][j], x_0))

            self.filter_array.append(filter_array_kari)
            
            
    def H_Filter_matome(self, x, r):
        
        y_kari = np.zeros((len(self.filter_array)))
        for i in range(len(self.filter_array)):
            y_kari[i] = x
            for j in range(len(self.filter_array[i])):
                y_kari[i] = self.filter_array[i][j].H_Filter(y_kari[i])
            
            y_kari[i] = y_kari[i] * r[i]
            
        y = 0
        for i in range(y_kari.shape[0]):
            y = y + y_kari[i] 
            
        return y
    
    
    
def filter_kakeru(x, r, time_array, a_array, b_array):
    y_new = np.zeros((0))

    count = 0
    for i in range(time_array.shape[0]):

        y_filter = x[int(count):int(count + time_array[i])].copy()
        r_filter = r[int(count):int(count + time_array[i])].copy()
        count = count + time_array[i]
        y0 = y_filter[0]
        
        filter_class = Digital_filter_matome_y_hozon(a_array, b_array, y0)
        for j in range(int(time_array[i])):
            y_filter[j] = filter_class.H_Filter_matome(y_filter[j], r_filter[j])
            
        y_new = np.hstack([y_new, y_filter])
        
    return y_new