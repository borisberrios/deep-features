from sklearn.manifold import TSNE
import os
import sys
import argparse
import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
import utils

class SktTSNE:
    """
    basado en: https://github.com/kylemcdonald/Coloring-t-SNE

    """

    def __init__(self, features, figsize=(5,5)):
        self.features = features
        self.figsize = figsize

    def fit(self):
        tsne = TSNE(n_components=2, random_state=0)
        self.sketches_tsne = tsne.fit_transform(self.features)


    def plot_tsne(self, xy, colors=None, alpha=0.7, s=3, cmap='hsv'):
        plt.figure(figsize=self.figsize, facecolor='white')
        plt.margins(0)
        plt.axis('off')
        fig = plt.scatter(xy[:,0], xy[:,1],
                    c=colors, # set colors of markers
                    cmap=cmap, # set color map of markers
                    alpha=alpha, # set alpha of markers
                    marker='o', # use smallest available marker (square)
                   # s=s, # set marker size. single pixel is 0.5 on retina, 1.0 otherwise
                  #  lw=0, # don't use edges
                    edgecolor='') # don't use edges
        # remove all axes and whitespace / borders
        fig.axes.get_xaxis().set_visible(True)
        fig.axes.get_yaxis().set_visible(True)
        plt.show()

    def show_standar(self):
        self.plot_tsne(self.sketches_tsne)

    def show_with_color(self):
        nns = NearestNeighbors(n_neighbors=10).fit(self.sketches_tsne)
        distances, indices = nns.kneighbors(self.sketches_tsne)

        distances = []
        for point, neighbor_indices in zip(self.features, indices):
            neighbor_points = self.features[neighbor_indices[1:]] # skip the first one, which should be itself
            cur_distances = np.sum([euclidean(point, neighbor) for neighbor in neighbor_points])
            distances.append(cur_distances)
        distances = np.asarray(distances)
        distances -= distances.min()
        distances /= distances.max()

        self.plot_tsne(self.sketches_tsne, np.clip(distances, 0, 1), cmap='viridis')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-filedir', type = str, required = True, help = 'directorio de arreglos deep features y labels')
    parser.add_argument('-colors', help = 'flag que indica la utilizacion de square root normalizacion', action='store_true')

    args = parser.parse_args()
    filedir = args.filedir
    colors = args.colors

    if not os.path.exists(filedir):
        sys.exit(1)

    deep_features, query_deep_features = utils.load_deep_features(filedir)
    skttsne = SktTSNE(query_deep_features)

    print ("Generando vectores T-SNE")
    skttsne.fit()

    if colors:
        skttsne.show_with_color()
    else:
        skttsne.show_standar()
