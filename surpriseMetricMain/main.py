import dataset
import closeness
import perplexity
import demo

import numpy as np
import matplotlib.pyplot as plt


def getNewAnnotations(image):
    d = dataset.Dataset()
    print(d.get_all_data().get(image))


def surprise(previous_annotations, new_annotations):
    close = closeness.Closeness(previous_annotations)
    perp = perplexity.Perplexity(previous_annotations)

    c_scores = []
    p_scores = []

    for a in new_annotations:
        # print(a)
        c_score = close.score(a)
        c_scores.append(c_score)

        p_score = perp.score(a)
        p_scores.append(p_score)

    return [c_scores, p_scores]


def normalise(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    combined = np.hstack((array1, array2))
    minimum = combined.min()
    maximum = combined.max()
    norm_array1 = (array1 - minimum) / (maximum - minimum)
    norm_array2 = (array2 - minimum) / (maximum - minimum)
    return [norm_array1, norm_array2]


def graph(correct_c_scores, incorrect_c_scores, correct_p_scores, incorrect_p_scores):
    plt.figure()
    plt.hlines(1, 1, 20)  # Draw a horizontal line
    plt.xlim(0, 1)
    plt.ylim(0.5, 1.5)

    y1 = np.ones(np.shape(correct_c_scores))  # Make all y values the same
    y2 = np.ones(np.shape(incorrect_c_scores))
    plt.plot(correct_c_scores, y1, '|', ms=40, label="Correct annotation")
    plt.plot(incorrect_c_scores, y2, '|', ms=40, label = "Incorrect annotation")
    plt.legend(markerscale=0.2)
    plt.axis('off')
    plt.title('Closeness scores plotted in range 0-1')
    plt.savefig('closeness.png')

    plt.figure()
    plt.hlines(1, 1, 20)  # Draw a horizontal line
    plt.xlim(0, 1)
    plt.ylim(0.5, 1.5)

    y1 = np.ones(np.shape(correct_p_scores))  # Make all y values the same
    y2 = np.ones(np.shape(incorrect_p_scores))
    plt.plot(correct_p_scores, y1, '|', ms=40, label="Correct annotation")
    plt.plot(incorrect_p_scores, y2, '|', ms=40, label="Incorrect annotation")
    plt.legend(markerscale=0.2)
    plt.axis('off')
    plt.title('Perplexity scores plotted in range 0-1')
    plt.savefig('perplexity.png')


def calculateScores(previous_annotations, correct_annotations, incorrect_annotations):
    correct_scores = surprise(previous_annotations, correct_annotations)
    incorrect_scores = surprise(previous_annotations, incorrect_annotations)

    correct_c_scores = correct_scores[0]
    incorrect_c_scores = incorrect_scores[0]
    c_scores = normalise(correct_c_scores, incorrect_c_scores)

    correct_c_scores = c_scores[0]
    incorrect_c_scores = c_scores[1]

    correct_p_scores = correct_scores[1]
    incorrect_p_scores = incorrect_scores[1]
    p_scores = normalise(correct_p_scores, incorrect_p_scores)

    correct_p_scores = p_scores[0]
    incorrect_p_scores = p_scores[1]

    return correct_c_scores, incorrect_c_scores, correct_p_scores, incorrect_p_scores


# gets 2 lots of saved annotations for different images from file
def getAnnotations():
    annotations = demo.get_test_annotations()

    # takes first 20 correct to test on
    correct_annotations = annotations[1][0:21]
    # uses the rest of the correct to train the language models
    previous_annotations = annotations[1][21:-1]

    # takes 20 incorrect annotations to test with
    incorrect_annotations = annotations[0][0:21]
    return previous_annotations, correct_annotations, incorrect_annotations


def getScores():
    return demo.get_scores()


def tunedScores(correct_c_scores, incorrect_c_scores, correct_p_scores, incorrect_p_scores):
    correct_scores = correct_c_scores + correct_p_scores * 0.001
    incorrect_scores = incorrect_c_scores + incorrect_p_scores * 0.001
    return correct_scores, incorrect_scores


scores = calculateScores(getAnnotations()[0], getAnnotations()[1], getAnnotations()[2])
# scores = getScores()
# tuned_scores = tune(scores[0], scores[1], scores[2], scores[3])
graph(scores[0], scores[1], scores[2], scores[3])
