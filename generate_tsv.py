import utils
import os
import argparse
import numpy as np


def generate_feature_tsv(base_file, features):
    tsv_file = open(os.path.join(base_file,"query_deep_feature.tsv"), "w")

    for feature in features:
        for i in range(len(feature)):
            tsv_file.write(str(feature[i]) + "\t")
        tsv_file.write("\n")
        tsv_file.flush()

    tsv_file.close()

def generate_labels_tsv(base_file, labels):
    labels_file = open(os.path.join(base_file, "query_deep_feature_labels.tsv"), "w")
    test_list = open(os.path.join(base_file, "test.txt"), "r")

    names = []
    for line in test_list:
        name, _ = line.split("\t")
        names.append(name)
    test_list.close()


    test_filename = os.path.join(base_file, "classes.txt")

    map = {}
    with open(test_filename) as test_in:
        for line in test_in:
            _line = line.split("\t")
            map[int(_line[1].strip())] = _line[0].strip()


    labels_file.write("id" + "\t" + "clase" + "\t" + "archivo" + "\n")
    for i in range(len(labels)):
        label = labels[i]
        labels_file.write(str(label) + "\t" + map[label] +  "\t" + names[i] +  "\n")

    labels_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-filedir', type = str, required = True, help = 'directorio de arreglos deep features y labels')

    args = parser.parse_args()
    file_base_dir = args.filedir

    deep_features, query_deep_features = utils.load_deep_features(file_base_dir)
    labs_train, labs_test = utils.np_labs_from_file(file_base_dir)

    generate_feature_tsv(file_base_dir, query_deep_features)
    generate_labels_tsv(file_base_dir, labs_test)
