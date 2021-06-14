from utils.utils import load_dict_map, load_item_profiles, standarize
from utils.plotter import Plotter
from recommendation.similarity_measures import CosineSimilarity
from recommendation.recommender import Recommender, RecommenderFilter
from models.item_profile import ItemProfileInceptionV3
from models.user_profile import ConstantDecayUserProfile
from config import EMBEDDING_SIZE

import streamlit as st
import PIL
import numpy as np

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
def load_model():
    ip = ItemProfileInceptionV3()
    return ip

@st.cache
def load_image(image_file):
	img = PIL.Image.open(image_file)
	return img 

@st.cache
def get_plotter():
    plt = Plotter(figsize=(10,10))
    return plt

@st.cache
def get_random_up():
    # create random user_prof
    user_prof = standarize(np.random.rand(EMBEDDING_SIZE))
    np.save('temp/user_profile.npy', user_prof)
    np.save('temp/user_filter.npy', [])
    
@st.cache
def get_filter():
    return RecommenderFilter()

@st.cache
def load_image(image_file):
	img = PIL.Image.open(image_file)
	return img 

# load important models
rec, df_map = load_recommender()

# cache set random vector user_profile
get_random_up()

# load item profile and plotter
ip = load_model()
plt = get_plotter()

if st.sidebar.button('Create new user'):
    # create random user_prof
    user_prof = standarize(np.random.rand(EMBEDDING_SIZE))
    np.save('temp/user_profile.npy', user_prof)
    np.save('temp/user_filter.npy', [])

# load user profile
user_prof =  np.load('temp/user_profile.npy')
up = ConstantDecayUserProfile(user_prof)

# load filtering
filt = RecommenderFilter()
filt.already_showed = list(np.load('temp/user_filter.npy'))

# title
st.sidebar.title('Product recommender App')

# get similar index
user_profile = up.get()
index = rec.get(user_profile)

# apply filter
index = filt.apply(index, n=5)

# display first recommended item
img = load_image(df_map[index[0]])
st.image(img, width=300)

# plot other recommended items
img = plt.plot_items([df_map[i] for i in index[1:]])
st.pyplot(img)

# slider to select feedback
feedback = st.sidebar.select_slider('Slide to select', options=[-1,0,1])

if st.sidebar.button('Update'):
    # update profile with feedback
    up.update(rec.item_profiles[index[0]],reaction=feedback)
    # save new user profile and filter
    np.save('temp/user_profile.npy', up.get())
    np.save('temp/user_filter.npy', filt.already_showed)
    # rerun script
    st.experimental_rerun() 