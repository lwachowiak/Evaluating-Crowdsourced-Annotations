import dataset
import closeness
import perplexity
import demo
import classifier

import numpy as np
import matplotlib.pyplot as plt


def getNewAnnotations(image):
    d = dataset.Dataset()
    print(d.get_all_data().get(image))


def showImage(image_id):
    d = dataset.Dataset()
    d.visualise(image_id)


def normalise(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    combined = np.concatenate((array1, array2))
    minimum = combined.min()
    maximum = combined.max()
    norm_array1 = (array1 - minimum) / (maximum - minimum)
    norm_array2 = (array2 - minimum) / (maximum - minimum)
    return [norm_array1, norm_array2]


def calculateCScores(previous_annotations, new_annotations, center):
    if center is not None:
        close = closeness.Closeness(center)
        c_scores = []
        for a in new_annotations:
            # print(a)
            c_score = close.score(a)
            c_scores.append(c_score)
    else:
        close = closeness.Closeness(None)
        close.initialise(previous_annotations)
        c_scores = []
        for a in new_annotations:
            # print(a)
            c_score = close.score(a)
            c_scores.append(c_score)

    return c_scores


def calculatePScores(previous_annotations, new_annotations):
    perp = perplexity.Perplexity(previous_annotations)
    p_scores = []
    for a in new_annotations:
        p_score = perp.score(a)
        p_scores.append(p_score)

    return p_scores


def calculateScores(previous_annotations, correct_annotations, incorrect_annotations):
    correct_c_scores = calculateCScores(previous_annotations, correct_annotations)
    incorrect_c_scores = calculateCScores(previous_annotations, incorrect_annotations)
    #    c_scores = normalise(correct_c_scores, incorrect_c_scores)

    #    correct_c_scores = c_scores[0]
    #   incorrect_c_scores = c_scores[1]

    correct_p_scores = calculatePScores(previous_annotations, correct_annotations)
    incorrect_p_scores = calculatePScores(previous_annotations, incorrect_annotations)
    #    p_scores = normalise(correct_p_scores, incorrect_p_scores)

    #   correct_p_scores = p_scores[0]
    #   incorrect_p_scores = p_scores[1]

    return correct_c_scores, incorrect_c_scores, correct_p_scores, incorrect_p_scores


# gets 2 lots of saved annotations for different images from file
def getAnnotations():
    annotations = demo.get_test_annotations()

    # takes first 20 correct to test on
    correct_annotations = annotations[0][0:21]
    # uses the rest of the correct to train the language models
    previous_annotations = annotations[0][21:-1]

    # takes 20 incorrect annotations to test with
    incorrect_annotations = annotations[1][0:21]
    return previous_annotations, correct_annotations, incorrect_annotations


def getScores():
    return demo.get_scores()


def getCenter():
    return demo.get_center()


def graph(correct_c_scores, incorrect_c_scores, correct_p_scores, incorrect_p_scores):
    plt.figure()
    plt.hlines(1, 1, 20)  # Draw a horizontal line
    plt.xlim(0, 1)

    y1 = np.ones(np.shape(correct_c_scores))  # Make all y values the same
    y2 = np.ones(np.shape(incorrect_c_scores))
    plt.plot(correct_c_scores, y1, '|', ms=40, label="Correct annotation")
    plt.plot(incorrect_c_scores, y2, '|', ms=40, label="Incorrect annotation")
    plt.legend(markerscale=0.2)
    plt.axis('off')
    plt.title('Closeness scores plotted in range 0-1')
    plt.savefig('closeness.png')

    plt.figure()
    plt.hlines(1, 1, 20)  # Draw a horizontal line
    plt.xlim(0, 1)

    y1 = np.ones(np.shape(correct_p_scores))  # Make all y values the same
    y2 = np.ones(np.shape(incorrect_p_scores))
    plt.plot(correct_p_scores, y1, '|', ms=40, label="Correct annotation")
    plt.plot(incorrect_p_scores, y2, '|', ms=40, label="Incorrect annotation")
    plt.legend(markerscale=0.2)
    plt.axis('off')
    plt.title('Perplexity scores plotted in range 0-1')
    plt.savefig('perplexity.png')


def trainAndSaveClassifier(c1, correct, incorrect):
    c1.trainAndSaveClassifier(correct, incorrect)


def demonstration(c, annotation):
    a = [annotation]
    print(annotation)
    previous_annotations = getAnnotations()[0]
    c_score = calculateCScores(previous_annotations, a, getCenter())[0]
    #    p_score = calculatePScores(previous_annotations, a)[0]
    print("Closeness score calculated: ", c_score)
    #    print("Proximity Score calculated: ", p_score)
    classification = c.classify(c_score)[0]
    #    print(classification)
    if classification == 0:
        print("Close to existing annotations, no need to review")
    else:
        print("Different from existing annotations, flagged for review")

    print()


def giveDemo(c):
    stop = False
    while not stop:
        a = input("Make a new annotation: ")
        print("Verifying annotation...")
        if a != "stop":
            demonstration(c, a)
        else:
            stop = True


# scores = calculateScores(getAnnotations()[0], getAnnotations()[1], getAnnotations()[2])
# print(scores)
scores = getScores()
# tuned_scores = tune(scores[0], scores[1], scores[2], scores[3])
# graph(scores[0], scores[1], scores[2], scores[3])

# getNewAnnotations(('trains', 61602))


c1 = classifier.Classifier()
# trainAndSaveClassifier(c1, scores[0], scores[1])
c1.loadClassifier()

giveDemo(c1)
