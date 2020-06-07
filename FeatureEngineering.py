import os
import numpy as np
from PIL import Image
from skimage.feature import hog
import sklearn.preprocessing as pp
import more_itertools as mit


class FeatureEngineering:
    def __init__(self):
        self.training_set = [[] for _ in range(26)]
        self.test_set = [[] for _ in range(26)]

    def HOG(self):
        """ Loads images from the dataset and applies pre-processing
        using Histogram of Oriented Gradients """

        # root, character-directories, images
        for r, c, i in os.walk(os.getcwd() + "\dataset\chars74k-lite"):
            for img in i:
                if img == 'LICENSE':
                    continue

                pic = hog(np.array(Image.open(r + "\\" + img)),
                    orientations=4, pixels_per_cell=(2,2), cells_per_block=(1,1))

                pic_array = np.reshape(pic, 400)     # 20x20
                self.training_set[ord(img[0])-97] += [pic_array]    # Int of the Unicode char

        for letter in self.training_set:
            letter = np.vectorize(lambda x: x/255.0)(letter)

    def scaling(self):
        """ Preprocessing using scaling """

        for letter in self.training_set:
            for img in letter:
                img = pp.scale(img)     # Get unit variance and 0 mean

    def splitDataset(self):
        """ Gets data for both training and test classification """

        for i in range(26):
            size = len(self.training_set[i])-1
            samples = mit.random_combination(range(size,-1,-1), r=round(size*0.2))
            for s in samples:
                self.test_set[i] += [self.training_set[i][s]]
                del self.training_set[i][s]


if __name__ == "__main__":
    print("\t--- Feature Engineering ---\n")

    fe = FeatureEngineering()
    fe.HOG()
    fe.scaling()
    fe.splitDataset()
