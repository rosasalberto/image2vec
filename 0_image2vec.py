from models.item_profile import ItemProfileInceptionV3
from utils.utils import get_filepaths, save_item_profiles, save_dict_map, splitting
from config import EMBEDDING_SIZE, N_IMG, DATABASE_PATH, DO_KPCA

from sklearn.decomposition import KernelPCA
import numpy as np
from tqdm import tqdm

from pickle import dump, load

# how to get filepaths from 1 folder
filepaths = get_filepaths(DATABASE_PATH, n_img=N_IMG)

# create item profile model
ip = ItemProfileInceptionV3()

# Split to get all item profiles and store in numpy array
splits = splitting(filepaths, 100)

item_profiles = []
for i in tqdm(range(len(splits))):
    # get profile of split
    X = ip.get(splits[i])
    
    # save item profiles
    item_profiles.append(X)

item_profiles = np.append(np.array(item_profiles[:-1]).reshape(-1,2048),np.array(item_profiles[-1]).reshape(-1,2048), axis=0)

if DO_KPCA:
    # run kernel PCA
    # vector of size EMBEDDING_SIZE for item profiles
    transformer = KernelPCA(n_components=EMBEDDING_SIZE, kernel='linear')
    item_profiles = transformer.fit_transform(item_profiles)

    # save and load the scaler examples
    dump(transformer, open('./models/transformer.pkl', 'wb'))

# save database to specific path
database_path = './profiles/item_profiles.npy'
save_item_profiles(item_profiles, database_path)

# save dict to specific path
dict_path = './profiles/df_map.pkl'
save_dict_map(filepaths, dict_path)