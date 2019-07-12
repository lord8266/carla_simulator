import numpy as np
import matplotlib.pyplot as plt
batch_size =200

def get_len(data):
    cnt =0
    for i in range(len(data)-1):
        if data[i]==0 and data[i+1]==0 and data[i+2]==0:
            break
        cnt+=1
    
    return cnt

avg = np.load('reward_data_avg.npy')
avg = avg[avg!=0]
min_ = np.load('reward_data_min.npy')
min_ = min_[min_!=0]
max_ = np.load('reward_data_max.npy')
max_ = max_[max_!=0]

plt.plot(np.arange(get_len(avg))*batch_size,avg[:get_len(avg)],label='avg')
plt.plot(np.arange(get_len(avg))*batch_size,min_[:get_len(avg)],label='min')
plt.plot(np.arange(get_len(avg))*batch_size,max_[:get_len(avg)],label='max')
plt.legend(loc='lower right')
plt.show()
