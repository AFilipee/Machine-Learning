import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from SVMclassifier import SVM, runClassifier
import matplotlib.pyplot as plt

class Detection:
    def __init__(self):
        """ Loads and processes images from the detection dataset """

        self.imgs = [[],[]]
        self.predictions = [[],[]]

        # root, directories, images
        '''for r, d, i in os.walk(os.getcwd() + "\dataset\detection-images"):
            
            pic1 = hog(np.array(Image.open(r + "" + i[0])),
                    orientations=4, pixels_per_cell=(2,2), cells_per_block=(1,1))
            pic2 = hog(np.array(Image.open(r + "" + i[1])),
                    orientations=4, pixels_per_cell=(2,2), cells_per_block=(1,1))'''

        for i in (0,1):
            self.imgs[i-1] = Image.open(".\dataset\detection-images\detection-" + str(i+1) + ".jpg")

        #self.imgs = [np.reshape(pic1, len(pic1)), np.reshape(pic2, len(pic2))]   # 20x20

    def getPredictions(self):
        return self.predictions

    def getImgs(self):
        return self.imgs

    def scan(self, window_size, stride, detect_score_threshold):
        """ Determines predictions of letters and their positions, according to the size of
          the sliding windows, step of the sliding window for scanning and the min
          detection score to be considered a hit """

        num_pixels = window_size[0]*window_size[1]

        for i in (0,1):
            for s in self.slidingWindow(self.imgs[i], window_size, stride):
                detect_score = self.detect(s[2], num_pixels)
                if detect_score >= detect_score_threshold:
                    self.predictions[i] += [(s[0], s[1], detect_score)]

    def NMS(self, window_size, max_boxes, iou_threshold):
        """ Non-Maximum Suppression """

        for b in range(len(self.predictions)):
            detect_scores = np.array([box[2] for box in self.predictions[b]])
            boxes_coords = np.array([
                [box[0], box[1], box[0]+window_size[0], box[1]+window_size[1]] for box in self.predictions[b]])

            nms_inds = []
            indices = np.argsort(detect_scores)

            while len(indices) > 0 and len(nms_inds) < max_boxes:
                last = len(indices)-1
                ind_max = indices[last]
                nms_inds += [ind_max]
                suppress = [last]

                for i in range(last):
                    if self.iou(boxes_coords[ind_max], boxes_coords[indices[i]]) > iou_threshold:
                        suppress += [i]

                indices = np.delete(indices, suppress)

            # Use index arrays to select only nms_indices from boxes, scores, and classes
            self.predictions[b] = [
                (boxes_coords[index, 0], boxes_coords[index, 1], detect_scores[index])
                for index in nms_inds]



    def slidingWindow(self, img, window_size, stride):
        """ Slices predictions off from the image with the size
          of the window and the given stride """

        for x in range(0, img.size[0]-window_size[0], stride[0]):
            for y in range(0, img.size[1]-window_size[1], stride[1]):
                yield (x, y, img.crop((x, y, x+window_size[0], y+window_size[1])))

    def detect(self, crop, box_size):
        """ Calculates the ratio of non-white (<255) pixels to total pixels"""

        return np.array(crop)[np.array(crop) < 255].size / float(box_size)

    def iou(self, box1, box2):
        """ Intersection Over Union """

        # Coordinates of the intersection of the boxes
        x_min = max(box1[0], box2[0])
        y_min = max(box1[1], box2[1])
        x_max = min(box1[2], box2[2])
        y_max = min(box1[3], box2[3])

        # Area given by Union(A,B) = A+B-Inter(A,B)
        intr = 0
        if x_max > x_min and y_max > y_min:
            intr = float((x_max-x_min)*(y_max-y_min))
        union = float((box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - intr)

        return intr/union   # IOU

    def plot(self, classified_boxes, i, window_size):
        image = self.imgs[i]

        fig1 = plt.figure(dpi=400)
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.imshow(image)
        ax1.axis('off')
        for box in classified_boxes:
            x_min, y_min, x_max, y_max = box[0]-.5, box[1]-.5, box[0]+window_size[0]-.5, box[1]+window_size[1]-.5
            prediction = chr(int(box[2])+97)
            ax1.text(x_min, y_min-3, "%s" % prediction, color="red", fontsize=3)
            x = [x_max, x_max, x_min, x_min, x_max]
            y = [y_max, y_min, y_min, y_max, y_max]
            line, = ax1.plot(x,y,color="red")
            line.set_linewidth(.5)
        fig1.savefig("classification-" + str(i) + ".png")
        #plt.show()


if __name__ == "__main__":

    model = runClassifier()

    print("\t--- Detection ---\n")

    window_size = (20, 20)
    detn = Detection()
    detn.scan(window_size, (1,1), .85)
    detn.NMS(window_size, 100, .1)

    classified_boxes1 = model.classifyDetection(detn.getImgs()[0], detn.getPredictions()[0], window_size)
    classified_boxes2 = model.classifyDetection(detn.getImgs()[1], detn.getPredictions()[1], window_size)

    detn.plot(classified_boxes1, 0, window_size)
    detn.plot(classified_boxes2, 1, window_size)

