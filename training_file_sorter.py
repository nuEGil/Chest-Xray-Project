import os 
import pandas as pd 
from helper_functions import *

# given a finding label , sample n images starting at start
def filter_and_sample(x, finding, label =0, start = 0, n = 100):
    # restrict to a certain finding
    subx = x[x['Finding Labels'] == finding]

    # collect a group of samples
    subx = subx.iloc[start : start + n, :]
    subx['label'] = label
    
    return subx

# combine 2 finding files into 1 csv and save it.
def combine_2(data_, find0 = 'No Finding', find1 = 'Pneumonia', n = 50, start = 0, tag='train'):
    # make training data csv 
    combo = [filter_and_sample(data_, find0, label = 0, start = start, n = n),
                    filter_and_sample(data_, find1, label = 1, start = start, n = n)]

    combo = pd.concat(combo, axis = 0, ignore_index = True)
    combo = combo.sample(frac = 1.0)
    combo.to_csv('../meta_data_files/{}.csv'.format(tag))
    return combo

# manager function to organize things. 
def start():
    # read the data file 
    data_ = pd.read_csv('../Data_Entry_fullpath_2017.csv')
    
    # combine findings
    train_data = combine_2(data_, find0 = 'No Finding', find1 = 'Pneumonia', n = 50, start = 0, tag = 'train')
    test_data = combine_2(data_, find0 = 'No Finding', find1 = 'Pneumonia', n = 25, start = 100, tag = 'test')
    
    # print samples and shapes
    print(train_data.head())
    print(train_data.shape, test_data.shape)
     
if __name__ == '__main__':
    start()