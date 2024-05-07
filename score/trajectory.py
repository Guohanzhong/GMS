import torch
import os
import numpy as np
import pandas as pd

L = np.array([[1e10,1,2,3,4,5],[1e10,1e10,0.8,2,4,4.2],[1e10,1e10,1e10,2.3,4,6],[1e10,1e10,1e10,1e10,16,20],[1e10,1e10,1e10,1e10,1e10,11],[1e10,1e10,1e10,1e10,1e10,1e10]])
#print(L[0,1])
# L(s,t): s -> t 
#L = L.T
T = L.shape[0]-1
#D = torch.ones(size=(T+1,T+1))*torch.tensor(-1)
D = np.full([T+1,T+1], -1)
C = np.full([T+1,T+1], 1e10)
C[0,0] = 0
C[0,1:] = 1e10
for k in range(1,T+1):
    for t in range(1,T+1):
        C[k,t] = np.min(C[k-1,:]+L[:,t])
        D[k,t] = np.argmin(C[k-1,:]+L[:,t])
    #bpds = C[k-1,None]+L[:,:]
    #C[k] = np.min(bpds,axis=-1)
    #D[k] = np.argmin(bpds,axis=-1)
#print(C)
#print(T)
steps = 3
init_steps = D[steps,T]
temp_list = [T]
for ele in range(steps):
    next_states = D[steps-ele,temp_list[ele]]
    #next_states = 
    temp_list.append(next_states)
    #steps = next_states
#print(temp_list)
#print(C)
#print(D)
#C = torch.ones(size=(T+1,T+1))*torch.tensor(-1)
#print(torch.tensor(C))
path_file = '/home/aiops/allanguo/cifar/merror/'
file_list = os.listdir(path_file)

data = pd.read_csv('/home/aiops/allanguo/cifar/merror/error_f.csv',index_col=0)
print(data)



KK = 999
kk = 73
steps = 15
data = data.iloc[kk:KK+1,kk:KK+1]
print(data)

ratio = (KK-kk)/steps

T = KK-kk
final_data = np.full([T+1,T+1], 1e10)
print(final_data.shape)
#print(data)
for i in range(T+1):
    for j in range(i+1,T+1):
        final_data[i,j] = data.iloc[i,j]
#print(final_data)

T = final_data.shape[0]-1
D = np.full([T+1,T+1], -1)
C = np.full([T+1,T+1], 1e10)
C[0,0] = 0
C[0,1:] = 1e10
for k in range(1,T+1):
    for t in range(1,T+1):
        D_temp = np.full([T+1,T+1], 1e10)
        #if t+1-k <= 3:
        #    D_temp[t-int(ratio*3):,t] = final_data[t-int(ratio*3):,t]
        #else:
        #D_temp[t-int(ratio*(k/2)):,t] = final_data[t-int(ratio*(k/2)):,t]
        #D_temp[t-int(ratio*1.2):,t] = final_data[t-int(ratio*1.2):,t]
        #C[k,t] = np.min(C[k-1,:]+D_temp[:,t])
        C[k,t] = np.min(C[k-1,:]+(k)**(0.5)*final_data[:,t])
        #D[k,t] = np.argmin(C[k-1,:]+D_temp[:,t])
        D[k,t] = np.argmin(C[k-1,:]+(k)**(0.5)*final_data[:,t])
    #bpds = C[k-1,None]+L[:,:]

init_steps = D[steps,T]
temp_list = [T+kk]
print(temp_list)
for ele in range(steps):
    next_states = D[steps-ele,temp_list[ele]-kk]
    #next_states = 
    temp_list.append(next_states+kk)
    #steps = next_states
print(temp_list)



#flist= [999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 212, 202, 192, 182, 172, 162, 153, 143, 134, 125, 116, 107, 99, 91, 83, 75, 67, 59, 52, 0]
#flist= [999, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 212, 202, 192, 182, 172, 162, 153, 143, 134, 125, 116, 107, 99, 91, 83, 75, 67, 59, 52,39,19, 0]
#flist= [999, 919, 879, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 349, 304, 278, 257, 239, 223, 209, 195, 181, 167, 153, 140, 127, 114, 101, 89, 77, 64, 50, 39, 19,0]
#flist= [999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719, 699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 499, 479, 459, 439, 419, 399, 379, 359, 339, 319, 299, 279, 259, 239, 219, 199, 179, 159, 139, 119, 99, 79, 59, 39, 19, 0]
#print(len(flist))
#print(D)
#[999, 919, 879, 839, 799, 759, 719, 679, 639, 599, 559, 519, 479, 439, 399, 349, 299, 249, 214, 186, 158, 128, 98, 50, 39, 0]
#[999, 959, 919, 879, 839, 799, 759, 719, 679, 639, 599, 559, 519, 479, 439, 399, 359, 319, 279, 239, 199, 159, 119, 79, 39, 0]
#[999, 799, 699, 599, 499, 399, 319, 169, 150,99, 0]
#[399, 319, 169, 150, 0]
#[999, 989, 979, 969, 959, 949, 939, 929, 919, 909, 899, 889, 879, 869, 859, 849, 839, 829, 819, 809, 799, 789, 779, 769, 759, 749, 739, 729, 719, 709, 699, 689, 679, 669, 659, 649, 639, 629, 619, 609, 599, 589, 579, 569, 559, 549, 539, 529, 519, 509, 499, 489, 479, 469, 459, 449, 439, 429, 419, 409, 399, 389, 379, 369, 359, 349, 339, 329, 319, 309, 299, 289, 279, 269, 259, 249, 239, 229, 219, 209, 199, 189, 179, 169, 159, 149, 139, 129, 119, 109, 99, 89, 79, 69, 59, 49, 39, 29, 19, 9, 0]
#[999, 959, 949, 939, 929, 919, 909, 899, 889, 879, 869, 859, 849, 839, 829, 819, 809, 799, 789, 779, 769, 759, 749, 739, 729, 719, 709, 699, 689, 679, 669, 659, 649, 639, 629, 619, 609, 599, 589, 579, 569, 559, 549, 539, 529, 519, 509, 499, 489, 479, 469, 459, 449, 439, 429, 419, 409, 399, 370, 348, 330, 313, 299, 286, 274, 263, 253, 244, 235, 227, 219, 211, 203, 195, 187, 179, 172, 164, 157, 149, 141, 134, 127, 120, 113, 106, 99,  93,  87,  81,  75, 69, 63, 57, 51, 45, 40, 29, 19, 9, 0]
    
    


