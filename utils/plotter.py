import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Plotter:
    def __init__(self, figsize):
        self.figsize = figsize
        
    def plot(self, filepaths):
        if len(filepaths) > 1:
            self.plot_items(filepaths)
        else:
            self.plot_item(filepaths[0])
    
    def plot_item(self, filepath):
        plt.figure(figsize=self.figsize)
        img = mpimg.imread(filepath)
        plt.axis('off')
        plt.imshow(img)
        return plt

    def plot_items(self, filepaths):
        fig, ax = plt.subplots(1,len(filepaths),figsize=self.figsize)
        for i in range(len(filepaths)):
            ax[i].axis('off')
            img = mpimg.imread(filepaths[i])
            ax[i].imshow(img);
        return fig