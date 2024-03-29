# https://www.geeksforgeeks.org/classifying-data-using-support-vector-machinessvms-in-python/

from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix



class Classifier:

    def __init__(self) -> None:
        self.classifier = None


    def trainAndSaveClassifier(self, correct_scores, incorrect_scores):
        x = correct_scores + incorrect_scores
        y = []
        for s in correct_scores:
            y.append(0)
        for s in incorrect_scores:
            y.append(1)

        x, x_test, y, y_test = train_test_split(np.array(x), np.array(y))
        self.classifier = SVC()
        self.classifier.fit(np.array(x), np.array(y))
        y_pred = self.classifier.predict(x_test)
        print("Accuracy of classifier: ", accuracy_score(y_test, y_pred))
        print("Confusion matrix: ", confusion_matrix(y_test, y_pred))

        pickle.dump(self.classifier, open("classifier.pkl", "wb"))


    def loadClassifier(self):
        self.classifier = pickle.load(open("classifier.pkl", "rb"))


    def classify(self, annotation_score):
        return self.classifier.predict(annotation_score)
