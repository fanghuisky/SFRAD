
import os
import numpy as np

used_number_samples = [1,2,5,10,16,20,50]
seed = [20,20,20,20,10,10,10]
dataset_path = '/home/zhy/anomaly/datasets/KSDD'
class_name = ['0','1','2']
dataset_name='ksdd'

save_label_ = open("{}_select_data.txt".format(dataset_name), "a+")
for i, K_ALL in zip(used_number_samples,seed):
    for k in range(K_ALL):
        for j in class_name:
            num_all = len(os.listdir('{}/{}/train/good/'.format(dataset_path,j)))
            random_number = np.random.rand(num_all)
            select_data = np.argsort(random_number)
            str_ = ''
            for l in range(i):
                str_ = str_ + '{} '.format(select_data[l])
            save_label_.write('{}\n'.format(str_))

save_label_.close()
