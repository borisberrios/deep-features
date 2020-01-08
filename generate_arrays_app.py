import Similarity
from Sketches import Sketches
from CNN import CNN
import os
import numpy as np
import math

"""
 Tener GPU para calcular los deep features
"""


def generate_array_images(dir_image_base, dir_files_base):
    """
        Transforma las imagenes y etiquetas a numpy array.
        las imagenes por defecto son (128,128)
     """

    sketches = Sketches(dir_image_base, dir_files_base)
    imgs_train, labs_train, _ = sketches.as_array("train.txt")
    imgs_test, labs_test, _ = sketches.as_array("test.txt")

    imgs_train.astype(np.float32).tofile(os.path.join(dir_files_base, "imgs_train.np"))
    labs_train.astype(np.int64).tofile(os.path.join(dir_files_base, "labs_train.np"))

    imgs_test.astype(np.float32).tofile(os.path.join(dir_files_base, "imgs_test.np"))
    labs_test.astype(np.int64).tofile(os.path.join(dir_files_base, "labs_test.np"))

    print (imgs_train.shape, labs_train.shape, imgs_test.shape, labs_test.shape)
    return imgs_train, labs_train, imgs_test, labs_test


def generate_deep_features(dir_model_base, file_base_dir, imgs_train, imgs_test):

    """
        Genera arreglo deep features
    """

    cnn_tf = CNN_TF(dir_model_base)

    deep_features = []
    query_deep_features = []

    batch_size = 1000
    for i in range( int(math.ceil(len(imgs_train) / batch_size)) ):
      current_deep_features = cnn_tf.features(imgs_train[i * batch_size: i * batch_size + batch_size])
      deep_features.extend(current_deep_features)
      print (f" deep features: {(i + 1) * batch_size}")

    deep_features = np.array(deep_features)

    for i in range( int(math.ceil(len(imgs_test) / batch_size)) ):
      current_deep_features = cnn_tf.features(imgs_test[i * batch_size: i * batch_size + batch_size])
      query_deep_features.extend(current_deep_features)
      print (f" query deep features: {(i + 1) * batch_size}")

    query_deep_features = np.array(query_deep_features)

    deep_features.astype(np.float32).tofile(os.path.join(file_base_dir, "deep_features.np"))
    query_deep_features.astype(np.float32).tofile(os.path.join(file_base_dir, "query_deep_features.np"))

    return deep_features, query_deep_features


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-filedir', type = str, required = True, help = 'directorio en donde se encuentran archivo con listado de imagenes y en donde se guardan las deep features')
    parser.add_argument('-imagesdir', type = str, required = True, help = 'directorio base donde se encuentran las imagenes')
    parser.add_argument('-base_model_dir', type = str, required = True, help = 'directorio base donde se encuentra el modelo entrenado')

    args = parser.parse_args()
    file_base_dir = args.filedir
    imagesdir = args.imagesdir
    base_model_dir = args.base_model_dir

    imgs_train, labs_train, imgs_test, labs_test = generate_array_images(imagesdir, file_base_dir)
    deep_features = generate_deep_features(base_model_dir, file_base_dir, imgs_train, imgs_test)
