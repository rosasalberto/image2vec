import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Similarity:
    """ Similarity base class """
    def calculate(self):
        """ 
        Returns a similiraty measure,
        the less the value, the more similar it is from user_profile
        """
        raise NotImplementedError
        
class EucledianDistance(Similarity):
    """ Eucledian Distance as similarity measure """
    def calculate(self, user_profile, item_profiles):
        return np.sum(np.square(np.array(user_profile) - item_profiles), axis=1)
        
class CosineSimilarity(Similarity):
    """ Cosine similarity """
    def calculate(self, user_profile, item_profiles):
        """ Negative Cosine Similarity, in this way the less the similar """
        return -cosine_similarity(user_profile.reshape(1,-1),item_profiles).reshape(-1)