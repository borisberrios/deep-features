import numpy as np
import os
import argparse
import sys
import utils

from Similarity import Similarity
from Sketches import Sketches
from CNN import CNN

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-filedir', type = str, required = True, help = 'directorio de arreglos deep features y labels')
    parser.add_argument('-ranking', type = int, required = True, help = 'Cantidad de elementos renkings para calculo de ')
    
    args = parser.parse_args()
    file_base_dir = args.filedir
    ranking = args.ranking

    if not os.path.exists(file_base_dir):
        print (f"Directorio indicado no existe: {file_base_dir}")
        sys.exit()

    if not os.path.exists(os.path.join(file_base_dir, utils.LABS_TRAIN)) or \
        not os.path.exists(os.path.join(file_base_dir, utils.LABS_TEST)) or \
        not os.path.exists(os.path.join(file_base_dir, utils.IMGS_TRAIN)) or \
        not os.path.exists(os.path.join(file_base_dir, utils.IMGS_TEST)):
        print (f" Algunos de los archivos no se encontraron: {utils.LABS_TRAIN, utils.LABS_TEST, utils.IMGS_TRAIN, utils.IMGS_TEST}")
        sys.exit()


    labs_train, labs_test = utils.np_labs_from_file(file_base_dir)
    deep_features, query_deep_features = utils.load_deep_features(file_base_dir)

    similarity = Similarity(deep_features, labs_train)
    mAP = similarity.mAP(query_deep_features, labs_test, top = ranking)
    print ("mAP", mAP)
