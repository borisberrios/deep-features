from sklearn.manifold import TSNE
import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
import utils

class SktTSNE:
    """
    basado en: https://github.com/kylemcdonald/Coloring-t-SNE

    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def fit(self):
        tsne = TSNE(n_components=2, random_state=0)
        self.sketches_tsne = tsne.fit_transform(self.features)


    def plot(self, size = 5):
        df = pd.DataFrame({'class': self.labels,
               'x': self.sketches_tsne[:,0],
               'y': self.sketches_tsne[:,1]})

        frame = pd.pivot_table(df, index=['class','x'], values=['y'], aggfunc=np.sum).reset_index()
        sns.lmplot(x='x' , y='y', data=frame, hue='class',palette='hls', fit_reg=False,size= size, aspect=5/3, legend = False, legend_out=False,scatter_kws={"s": 20})
        plt.show()

    def show_with_filters(self, class_list, size = 5):
        df = pd.DataFrame({'class': self.labels,
               'x': self.sketches_tsne[:,0],
               'y': self.sketches_tsne[:,1]})

        df_scatter = df[df['class'].isin(class_list)]

        frame = pd.pivot_table(df_scatter, index=['class','x'], values=['y'], aggfunc=np.sum).reset_index()
        sns.lmplot(x='x' , y='y', data=frame, hue='class',palette='hls', fit_reg=False,size = size, legend = True, legend_out=False,scatter_kws={"s": 20})
        plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-filedir', type = str, required = True, help = 'directorio de arreglos deep features y labels')
    parser.add_argument('-class_list', nargs='*', type=int, default = [])

    args = parser.parse_args()
    filedir = args.filedir
    class_list = args.class_list

    if not os.path.exists(filedir):
        sys.exit(1)

    #   print ("lista de clases:", class_list, len(class_list))

    deep_features, query_deep_features = utils.load_deep_features(filedir)
    labs_train, labs_test = utils.np_labs_from_file(filedir)

    skttsne = SktTSNE(query_deep_features, labs_train)

    print ("Generando vectores T-SNE")
    skttsne.fit()

    if len(class_list) > 0:
        skttsne.show_with_filters(class_list)
    else:
        skttsne.plot()
