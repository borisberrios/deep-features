import argparse
import numpy as np
import os
import utils
import matplotlib.pyplot as plt

from Sketches import Sketches
from Sketches import Images
from Similarity import Similarity
from CNN import CNN

class SktSearcher:

    def __init__(self, base_images_dir, base_files_dir, base_dir_model, to_normalize = False):
        self.base_images_dir = base_images_dir
        self.images = Images()
        self.base_files_dir = base_files_dir
        self.test_labels_map = self._generate_query_label_maps()
        self.test_name_label_map = self._generate_id_class_map()
        self.cnn = CNN(base_dir_model)

        self.labs_train, _ = utils.np_labs_from_file(base_files_dir)
        deep_features, _ = utils.load_deep_features(base_files_dir)

        self.similarity = Similarity(deep_features, self.labs_train)
        self.filenames_db = self._generate_list_filename_train()

    def _generate_list_filename_train(self):
        """ genera una lista con los nombres de los archivos de la db de imagenes """

        test_filename = os.path.join(self.base_files_dir, "train.txt")
        filanames_db = []
        with open(test_filename) as test_in:

            for line in test_in:
                _line = line.split("\t")
                filanames_db.append(_line[0].strip())

        return filanames_db

    def _generate_query_label_maps(self):
        """ Se genera dict que mapea nombre de imagen query con id de etiqueta """

        test_filename = os.path.join(self.base_files_dir, "test.txt")
        map = {}
        with open(test_filename) as test_in:
            for line in test_in:
                _line = line.split("\t")
                map[_line[0].strip()] = int(_line[1].strip())
        return map

    def _generate_id_class_map(self):
        """ Se genera dict que mapea id de etiqueta con su nombre respectivo """

        test_filename = os.path.join(self.base_files_dir, "classes.txt")
        map = {}
        with open(test_filename) as test_in:
            for line in test_in:
                _line = line.split("\t")
                map[int(_line[1].strip())] = _line[0].strip()
        return map


    def _show_top(self, ranking, filename, filenames, labels):

        fig, xs = plt.subplots(1, len(ranking) + 1)
        for i in range(len(ranking) + 1):
            xs[i].set_axis_off()
        fig.canvas.set_window_title('Busqueda por similitud')

        myimg = self.images.read_image(filename)
        xs[0].imshow(myimg, 'gray')


        for i in range(len(ranking)):
            dis, pos = ranking[i]
            print (filenames[pos], dis, labels[pos])
            file = os.path.join(self.base_images_dir, filenames[pos])
            myimg = self.images.read_image(file)
            xs[i+1].imshow(myimg, 'gray')

        plt.tight_layout()
        plt.show()

    def run(self):

        print ("Ingrese imagen para busqueda por similutud. Ejemplos: airplane/33.png, alarm_clock/104.png, apple/391.png")
        print ("Para salir: exit")

        query = None

        while query != 'exit':
            print ("query>>> ", end = '')
            query = input()
            query = query.strip()

            if query != 'exit':
                filename = os.path.join(self.base_images_dir, query)
                if query in self.test_labels_map:
                    query_label = self.test_labels_map[query]
                    class_name = self.test_name_label_map[query_label]
                    print (f"imagen: {query}, label code: {query_label}, label name: {class_name}")

                    if os.path.exists(filename):
                        img = self.images.as_array_one(filename, (128,128))
                        fv = self.cnn.feature(img)
                        fv = np.squeeze(fv)
                        ranking = self.similarity.search(fv, to_normalize = True, top = 5)

                        ap = self.similarity.ap(ranking, self.labs_train, query_label)
                        print(f"ap: {ap}")

                        self._show_top(ranking, filename, self.filenames_db, self.labs_train)

                        """

                        ap = utils.ap(ranking, filenames, labels, query_label)
                        print(f"ap: {ap}")

                        if show_images:
                            utils.show_top(ranking, str_datadir, filename, filenames, labels)
                        else:
                            utils.list_ranking(ranking,filenames, labels)
                        """
                    else:
                        print (f"Archivo no existe: {query}")

                else:
                    print (f"La imagen no es parte del conjunto test: {query}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-filedir', type = str, required = True, help = 'directorio de arreglos deep features y labels')
    parser.add_argument('-imagesdir', type = str, required = True, help = 'directorio base donde se encuentran las imagenes')
    parser.add_argument('-base_model_dir', type = str, required = True, help = 'directorio base donde se encuentra el modelo entrenado')
    parser.add_argument('-ranking', type = int, required = True, help = 'Cantidad de elementos renkings para calculo de ')
    parser.add_argument('-normalization', help = 'flag que indica la utilizacion de square root normalizacion', action='store_true')

    args = parser.parse_args()
    file_base_dir = args.filedir
    ranking = args.ranking
    imagesdir = args.imagesdir
    base_model_dir = args.base_model_dir
    normalization = args.normalization

    if normalization:
        print ("Usando normalizacion")

    sktsearcher = SktSearcher(imagesdir, file_base_dir, base_model_dir, normalization)
    sktsearcher.run()
