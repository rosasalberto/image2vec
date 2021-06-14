import numpy as np
import tensorflow as tf

class ItemProfile:
    # base class, has to implement preprocess method and model property
    def get(self, img_paths):
        pre_imgs = self.preprocess(img_paths)
        item_profiles = self.model.predict(pre_imgs)
        return item_profiles
        
    def preprocess(self, img_paths):
        raise NotImplementedError

class ItemProfileInceptionV3(ItemProfile):
    def __init__(self):
        self.create_base_model()
        
    def create_base_model(self):
        # Inception V3 model, obtain intermediate layer, last vector
        base_model = tf.keras.applications.InceptionV3(weights='imagenet')
        self.model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
        
    def preprocess(self, img_paths):
        if len(img_paths) > 1:
            x = np.array([np.array(tf.keras.preprocessing.image.load_img(imgpath, target_size=(299, 299))) for imgpath in img_paths])
            pre_imgs = tf.keras.applications.inception_v3.preprocess_input(x)
        else:
            x =  np.expand_dims(tf.keras.preprocessing.image.load_img(img_paths[0], target_size=(299, 299)),axis=0)
            pre_imgs = tf.keras.applications.inception_v3.preprocess_input(x)
        return pre_imgs
        
    def preprocess_image(self, img_array):
        x = np.expand_dims(img_array, axis=0)
        pre_imgs = tf.keras.applications.inception_v3.preprocess_input(x)
        return pre_imgs
        
    def get_from_array(self, img_array):
        pre_imgs = self.preprocess_image(img_array)
        item_profiles = self.model.predict(pre_imgs)
        return item_profiles