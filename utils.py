import os
import numpy as np


LABS_TRAIN = 'labs_train.np'
LABS_TEST = 'labs_test.np'
IMGS_TRAIN = 'imgs_train.np'
IMGS_TEST = 'imgs_test.np'
DEEP_FEATURES = 'deep_features.np'
QUERY_DEEP_FEATURES = 'query_deep_features.np'


def np_labs_from_file(dir_files_base):
    """
        Se retornan los arreglos que contienen las etiquetas de cada imagenes
        en el mismo orden en el que aparecen en los archivos
    """
    labs_train = np.fromfile(os.path.join(dir_files_base, LABS_TRAIN), dtype=np.int64)
    labs_train = np.reshape(labs_train, (14000,))

    labs_test = np.fromfile(os.path.join(dir_files_base, LABS_TEST), dtype=np.int64)
    labs_test = np.reshape(labs_test, (6000,))

    return labs_train, labs_test

def np_images_from_file(dir_files_base):
    """
        Se retorna arreglo con images desde archivo
    """

    imgs_train = np.fromfile(os.path.join(dir_files_base, IMGS_TRAIN), dtype=np.float32)
    imgs_train = np.reshape(imgs_train, (14000,128,128,1))

    imgs_test = np.fromfile(os.path.join(dir_files_base, IMGS_TEST), dtype=np.float32)
    imgs_test = np.reshape(imgs_test, (6000,128,128,1))

    return imgs_train, imgs_test


def load_deep_features(dir_files_base):
    """
        Se retonan los arreglos deep features, del conjunto de entrenamiento (14000 imagenes)
        y del conjunto de test (6000 imagenes). Las imagenes del conjunt test serán las queries
    """

    deep_features = np.fromfile(os.path.join(dir_files_base, DEEP_FEATURES), dtype=np.float32)
    deep_features = np.reshape(deep_features, (14000,1024))

    query_deep_features = np.fromfile(os.path.join(dir_files_base, QUERY_DEEP_FEATURES), dtype=np.float32)
    query_deep_features = np.reshape(query_deep_features, (6000,1024))

    return deep_features, query_deep_features
