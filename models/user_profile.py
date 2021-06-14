import numpy as np

class UserProfile:
    # has to implement self.user_prof property and update method
    def update(self, item_prof, reaction=0):
        raise NotImplementedError
        
    def get(self):
        return self.user_prof
        
class ExponentialDecayUserProfile(UserProfile):
    def __init__(self, user_prof, c_decay=20, c_strength=10):
        self.user_prof = user_prof
        self.coeff = lambda x: (1 - 1/(1 + np.exp(-x/c_decay))) / c_strength
        self.counter = 0
            
    def update(self, item_prof, reaction=0):
        self.user_prof = self.user_prof + reaction*self.coeff(self.counter)*item_prof
        self.counter += 1
        
class ConstantDecayUserProfile(ExponentialDecayUserProfile):      
    def update(self, item_prof, reaction=0):
        self.user_prof = self.user_prof + reaction*self.coeff(0)*item_prof