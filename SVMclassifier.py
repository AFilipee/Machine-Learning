import numpy as np
from sklearn import datasets, svm
from sklearn.metrics import log_loss
from FeatureEngineering import FeatureEngineering as FE
from skimage.feature import hog
import sklearn.preprocessing as pp

class SVM:

    def __init__(self,X_train,y_train,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run(self):
        
        self.loss = []
        kernel = 'poly'
        # fit the model
        if(kernel == 'poly'):
            #if(kernel == 'linear'):
                #clf = svm.SVC(kernel=kernel, gamma=0.01, probability=True)
            #else:
            self.clf = svm.SVC(kernel=kernel, gamma=1, degree=5, probability=True)
            self.clf.fit(self.X_train, self.y_train)
            
            
            pred = np.zeros((self.y_test.shape[0],26))
            counter = 0

            for i in range(len(self.X_test)):
                
                pred[i] = self.clf.predict_proba(self.X_test[i].reshape(1,-1))
                if(self.clf.predict(self.X_test[i].reshape(1,-1)) == self.y_test[i]):
                    counter += 1

            avg = counter / len(self.y_test)
            print(avg)

            self.loss.append(log_loss(self.y_test,pred))
            print("Final Loss for ", kernel, " kernel: ",log_loss(self.y_test,pred), "\n\n")
        return self.loss

    def classifyDetection(self, image, predictions, window_size):
        classified_boxes = []
        for box in predictions:
            x, y = box[0], box[1]
            prediction = self.predict(image.crop((x,y,x+window_size[0],y+window_size[1])))
            classified_boxes += [(x, y, prediction)]
        return classified_boxes


    def predict(self, img):
        pic = hog(np.array(img), orientations=4, pixels_per_cell=(2,2), cells_per_block=(1,1))
        pic_array = pp.scale(np.reshape(pic, 400))

        self.clf.predict_proba(pic_array.reshape(1,-1))
        return self.clf.predict(pic_array.reshape(1,-1))




                    
def runClassifier():
    print("\t--- Feature Engineering ---\n")

    fe = FE()
    fe.HOG()
    fe.scaling()
    fe.splitDataset()

    print("\t--- SVM Classifier ---\n")


    i = 0

    for chars in range(len(fe.training_set)):

        for samples in range(len(fe.training_set[chars])):
            i += 1

    X_train = np.zeros((i,400))
    y_train = np.zeros((i,))

    i = 0

    for chars in range(len(fe.test_set)):

        for samples in range(len(fe.test_set[chars])):
            i += 1

    X_test = np.zeros((i,400))
    y_test = np.zeros((i,))

    i = 0

    for chars in range(len(fe.training_set)):

        for samples in range(len(fe.training_set[chars])):
            X_train[i] = fe.training_set[chars][samples]

            y_train[i] = chars
            i += 1

    i = 0

    for chars in range(len(fe.test_set)):

        for samples in range(len(fe.test_set[chars])):
            X_test[i] = fe.test_set[chars][samples]

            y_test[i] = chars
            i += 1

    model = SVM(X_train,y_train,X_test,y_test)

    loss = model.run()

    return model

if __name__ == "__main__":
    runClassifier()
