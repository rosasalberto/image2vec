from utils.utils import load_dict_map, load_item_profiles
from utils.plotter import Plotter
from recommendation.similarity_measures import CosineSimilarity
from recommendation.recommender import Recommender, RecommenderFilter
from models.item_profile import ItemProfileInceptionV3
from config import DO_KPCA

import streamlit as st
import PIL
import numpy as np

from pickle import load

@st.cache(allow_output_mutation=True)
def load_recommender():
    database_path = './profiles/item_profiles.npy'
    dict_path = './profiles/df_map.pkl'
    
    # load item_profiles 
    item_prof = load_item_profiles(database_path)
    df_map = load_dict_map(dict_path)
    
    rec = Recommender(similarity_measure=CosineSimilarity(), item_profiles=item_prof)
    return rec, df_map
 
@st.cache 
def load_pca():
    transformer = load(open('./models/transformer.pkl', 'rb'))
    return transformer

@st.cache
def load_model():
    ip = ItemProfileInceptionV3()
    return ip

@st.cache
def load_image(image_file):
	img = PIL.Image.open(image_file)
	return img 
    
@st.cache
def get_initialize():
    np.save('temp/image_filter.npy', [])

@st.cache
def get_plotter():
    plt = Plotter(figsize=(10,10))
    return plt
    
# initialize
get_initialize()

# load important models
rec, df_map = load_recommender()

# load filtering
filt = RecommenderFilter()
filt.already_showed = list(np.load('temp/image_filter.npy'))

ip = load_model()
plt = get_plotter()
if DO_KPCA:
    transformer = load_pca()

# title
st.sidebar.title('Image similarity App')

# file uploader
image_file = st.sidebar.file_uploader("Upload Image",type=['png','jpeg','jpg'])
if image_file is not None:
    # print file details
    st.sidebar.write(image_file)
    
    # load image and resizing as do tf.keras by default
    img = load_image(image_file)
    resized_img = np.array(img.resize((299, 299), resample=PIL.Image.NEAREST))[:,:,:3]
    
    # calculate item prof
    item_profile = ip.get_from_array(resized_img)
    
    if DO_KPCA:
        # pca reduction
        item_profile = transformer.transform(item_profile)
    
    # display image
    st.image(img, width=200)
    
    st.write('Similar items:')
    
    # get index of the similar items
    index = rec.get(item_profile)
    # apply filter
    index = filt.apply(index, n=4)
    
    # plot recommended items
    img = plt.plot_items([df_map[i] for i in index])
    st.pyplot(img)