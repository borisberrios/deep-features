import numpy as np

class Similarity:

  def __init__(self, data_images, data_labs):
    self.data_images = data_images
    self.normalized_data_images = self.batch_sqrt_normalization(data_images)
    self.data_labs = data_labs

  def sqrt_normalization(self, feature):
    feature = np.sign(feature) * np.sqrt(np.abs(feature))
    norm = np.sqrt(np.sum(feature * feature));
    feature = feature / norm
    return feature


  def batch_sqrt_normalization(self, features):
    features = np.sign(features) * np.sqrt(np.abs(features))
    norms = np.sqrt(np.sum(features * features, axis = 1));
    norms = np.reshape(norms, (norms.shape[0], 1))
    features = features / norms
    return features

  def cos_distance(self, v, vectors):
      dot = np.sum(v * vectors, axis = 1)
      div = np.linalg.norm(v) * np.linalg.norm(vectors, axis = 1)
      div = np.reshape(div, (div.shape[0], 1))
      dot = np.reshape(dot, (dot.shape[0], 1))
      cos = dot / div

      #por si alguno se escapa de este rango en algunos decimales
      cos = np.clip(cos, -1, 1)

      distance = np.sqrt(2 - 2*cos)
      distance
      return distance

  def _funcion_caracteristica(self, ranking, labels, query_label):
      caracteristica_map = set()
      c = []
      for rank in ranking:
          distancia, posicion = rank
          if labels[posicion] == query_label:
              c.append(1)
          else:
              c.append(0)

      return c


  def _precision(self, f_caracteristica, i):
    sum = np.sum(f_caracteristica[:i+1])
    p = sum / (i + 1)

    p *= f_caracteristica[i]
    return p

  def ap(self, ranking, labels, query_label):
      f_caracteristica = self._funcion_caracteristica(ranking, labels, query_label)

      ap = 0
      for i in range(len(ranking)):
          p = self._precision(f_caracteristica, i)
          ap += p

      cardinalidad_r = np.sum(f_caracteristica)
      if cardinalidad_r > 0:
          ap /= cardinalidad_r
      else:
          ap = 0

      return ap


  def mAP(self, query_deep_features, query_labels, top = 5, to_normalize = False):
    mAP = 0
    for i in range(len(query_deep_features)):
      ranking = self.search(query_deep_features[i], to_normalize = to_normalize, top = top)
      ap = self.ap(ranking, self.data_labs, query_labels[i])
      mAP += ap

      num_queries = i + 1
      if (num_queries % 100 == 0):
        print (f"queries calculadas: {num_queries}, suma actual: {mAP}, mAP actual: {mAP/num_queries}")


    mAP /= len(query_deep_features)

    return mAP

  def search(self, image, to_normalize = False, top = 5):

    fv = image
    db_images = self.data_images

    if to_normalize:
      #print ("[Similarity] usando normalizacion")
      fv = self.sqrt_normalization(image)
      db_images = self.normalized_data_images

    distances = self.cos_distance(fv, db_images)
    sorted_index = np.argsort(distances, axis = 0)

    sorted_index = np.squeeze(sorted_index)
    ranking = []
    for i in range(top):
        ranking.append((distances[sorted_index[i]], sorted_index[i]))

    ranking = np.array(ranking)
    return ranking
