import os
import skimage.io as io
import numpy as np

import skimage.transform as transf


class Images:

    def toUINT8(self, image) :
          if image.dtype == np.float64 :
              image = image * 255
          elif image.dtype == np.uint16 :
              image = image >> 8
          image[image<0]=0
          image[image>255]=255
          image = image.astype(np.uint8, copy=False)
          return image

    def process_image(self, image, imsize):
          """
          imsize = (h,w)
          """
          #resize uses format (w,h)
          image_out = transf.resize(image, imsize)
          image_out = self.toUINT8(image_out)
          return image_out

    def read_image(self, filename):
          """ read_image using skimage """

          if not os.path.exists(filename):
              raise ValueError(filename + " does not exist!")

          image = io.imread(filename, as_gray = True)
          image = self.toUINT8(image)
          assert(len(image.shape) == 2)

          return image

    def show(self, myimg):
      plt.imshow(myimg[:,:,0], 'gray')
      plt.tight_layout()
      plt.show()


    def as_array_one(self, filename, image_shape, number_of_channels = 1):

        image = self.read_image(filename)
        image = self.process_image(image, (image_shape[0], image_shape[1]))
        image = np.reshape(image, (image_shape[0], image_shape[1], number_of_channels))
        image = image.astype(np.float32)

        return image


class Sketches:
  def __init__(self, images_dir, list_dir, image_shape = (128, 128)):
    self.images_dir = images_dir
    self.list_dir = list_dir
    self.image_shape = image_shape
    self.number_of_channels = 1
    self.images = Images()


  def as_array(self, file_list):
    """
    retorna en forma de arreglo todas las imagenes contenidas dentro del archivo indicado
    Parameters:
    file_list: string
      archivo con lista de imagenes y la clase a la que pertenece

    Returs:
      numpy array
        lista de imagenes representada como matriz de numeros
      numpy array
        lista de las clases a las que pertenece cada imagen retornada
      numpy array
        matriz promediode de las imagenes calculadas
    """

    images_list = []
    category_list = []
    with open(os.path.join(self.list_dir, file_list), 'r') as f:
      i = 0
      for line in f:
        filename, category = line.split("\t")
        filename = os.path.join(self.images_dir, filename)

        image = self.images.as_array_one(filename, self.image_shape)

        images_list.append(image)
        category_list.append(int(category))
        i += 1
        if i % 500 == 0:
          print (f"imagenes procesadas: {i}")

      images_list = np.array(images_list)
      category_list = np.array(category_list)
      mean = np.mean(images_list)

    return images_list, category_list, mean
