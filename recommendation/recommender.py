import numpy as np

class RecommenderFilter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.already_showed = []
        
    def apply(self, idx, n):
        index_filtered = []
        for i in range(len(idx)):
            if idx[i] not in self.already_showed:
                index_filtered.append(idx[i])
        self.already_showed.append(index_filtered[0])
        return np.array(index_filtered)[:n]
    
        
class Recommender:
    def __init__(self, similarity_measure, item_profiles):
        self.similarity_measure = similarity_measure
        #self.filtering = RecommenderFilter()
        
        self.item_profiles = item_profiles
        
    def get(self, profile):
        sim = self.similarity_measure.calculate(profile, self.item_profiles)
        idx = sim.argsort()
        return idx