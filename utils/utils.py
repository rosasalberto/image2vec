from os import listdir
from os.path import isfile, join
import pickle

import numpy as np

# standarize item profiles, so each item has mean 0 and std 1
standarize = lambda x: (x - np.mean(x)) / np.std(x)
# split list or array, into elements of size n
splitting = lambda a,n : np.split(a, np.arange(n, len(a), n))

def get_filepaths(img_folder, n_img=1000):
    # get all the filenames from a specific folde
    filepaths = [img_folder+'/'+f for f in listdir(img_folder)[:n_img] if isfile(join(img_folder, f))]
    return filepaths
    
def save_item_profiles(item_profiles, path):
    # store item profiles in a database, standarized (mean 0, std 1)
    item_prof = np.apply_along_axis(standarize, axis=1, arr=item_profiles)
    np.save(path, item_prof)
    
def load_item_profiles(path):
    return np.load(path)
    
def save_dict_map(filepaths, save_path):
    df_map = {i:filepaths[i] for i in range(len(filepaths))}
    
    a_file = open(save_path, "wb")
    pickle.dump(df_map, a_file)
    a_file.close()
    
def load_dict_map(path):
    a_file = open(path, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output