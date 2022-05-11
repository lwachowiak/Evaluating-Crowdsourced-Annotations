import dataset
import closeness
import perplexity
import demo
import classifier
import kbscores
import time

import numpy as np
import matplotlib.pyplot as plt


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


def graph(correct_scores, incorrect_scores, filename):
    plt.figure()
    plt.hlines(1, 1, 20)  # Draw a horizontal line
    plt.xlim(0, 1)

    y1 = np.ones(np.shape(correct_scores))  # Make all y values the same
    y2 = np.ones(np.shape(incorrect_scores))
    normal = normalise(correct_scores, incorrect_scores)
    plt.plot(normal[0], y1, '|', ms=40, label="Correct annotation")
    plt.plot(normal[1], y2, '|', ms=40, label="Incorrect annotation")
    plt.legend(markerscale=0.2)
    plt.axis('off')
    plt.title('Scores plotted in range 0-1')
    plt.savefig(filename)


def trainAndSaveClassifier(c1, correct, incorrect):
    c1.trainAndSaveClassifier(correct, incorrect)


def getNewAnnotations(image):
    d = dataset.Dataset()
    dictionary = d.get_all_data()
    a = dictionary.get(image)


# gets 2 lots of saved annotations for different images from file
def getAnnotations(dictionary, image):
    annotations = dictionary.get(image)

    # takes first 40 correct to test on
    new_annotations = annotations[0:41]
    # uses the rest of the correct to train the language models
    previous_annotations = annotations[41:-1]
 #   print("length: ", len(previous_annotations))

    return previous_annotations, new_annotations

def flatten(a):
    new_a = []
    for x in a:
        for y in x:
            new_a = np.append(new_a, y)
    return new_a.tolist

def calculateCScores(c, new_annotations):
    c_scores = []
    for a in new_annotations:
        c_score = c.score(a)
        c_scores.append(c_score)

    return c_scores

def setUp(image_class1, image_class2):
    print("Getting annotations for all images in class...")
    d = dataset.Dataset()
    dictionary = d.get_all_data()
    previous_annotations = []
    correct_train_annotations = []
    for key in dictionary:
        if key[0] is image_class1:
            annotations = getAnnotations(dictionary, key)
            previous_annotations.append(annotations[0])
            correct_train_annotations.append(annotations[1])
    p = np.array(previous_annotations, dtype=object).flatten().tolist()

    incorrect_train_annotations = []
    for key in dictionary:
        if key[0] is image_class2:
            annotations = getAnnotations(dictionary, key)
            incorrect_train_annotations.append(annotations[1])

    c_correct_scores = []
    c_incorrect_scores = []
    for i in range(len(p)):
        close = closeness.Closeness()
        close.initialise(p[i])
        c_correct_scores.append(calculateCScores(close, correct_train_annotations[i]))
        c_incorrect_scores.append(calculateCScores(close, incorrect_train_annotations[i]))

    c_correct_scores = np.array(c_correct_scores).flatten().tolist()
    c_incorrect_scores = np.array(c_incorrect_scores).flatten().tolist()

#    file = open("scores.txt", "w+")
#   file.writelines([str(scores[0]), str(scores[1]), str(scores[2]), str(scores[3]), str(scores[4])])
#    file.close()
#    print("Correct calculations:", correct_scores)
#    print("Incorrect calculations:", incorrect_scores)
    graph(c_correct_scores, c_incorrect_scores, "closeness.png")

    correct_keyword = image_class1
    incorrect_keyword = image_class1

    kb = kbscores.KBScores()
    kb_scores = kbScores(kb, correct_train_annotations, incorrect_train_annotations, correct_keyword, incorrect_keyword)
    kb_correct_scores = kb_scores[0]
    kb_incorrect_scores = kb_scores[1]

    graph(kb_correct_scores, kb_incorrect_scores, "kb.png")

    print("Correct C scores", c_correct_scores)
    print("Correct KB scores", kb_correct_scores)
    print("Correct C scores", c_incorrect_scores)
    print("Correct KB scores", kb_incorrect_scores)
    correct_scores = np.vstack((c_correct_scores, kb_correct_scores)).T
    incorrect_scores = np.vstack((c_incorrect_scores, kb_incorrect_scores)).T
    c1 = classifier.Classifier()
    print("Training classifier...")
    trainAndSaveClassifier(c1, correct_scores.tolist(), incorrect_scores.tolist())


def demonstration(c, dictionary, annotation, image):
    a = [annotation]
    previous_annotations = getAnnotations(dictionary, image)[0] + getAnnotations(dictionary, image)[1]

    close = closeness.Closeness()
    close.initialise(previous_annotations)
    c_score = calculateCScores(close, a)[0]

    print("Closeness score calculated: ", c_score)

    kb = kbscores.KBScores()
    kb_score = kb.get_annotation_score(annotation, image[0])
    print("Knowledge base score calculated: ", kb_score)

    combined_score = np.append(c_score, kb_score)
    classification = c.classify(np.array(combined_score).reshape(1, -1))
    #    print(classification)
    if classification == 0:
        print("Close to existing annotations, no need to review")
    else:
        print("Different from existing annotations, flagged for review")

    print("Perplexity score: ", calculatePerplexity(previous_annotations, annotation))
    print()


def giveDemo(c, image):
    dictionary = dataset.Dataset().get_all_data()
    showImage(image[1])
    stop = False
    while not stop:
        a = input("Give a new annotation for the image: ")
        if a != "stop":
            demonstration(c, dictionary, a, image)
        else:
            stop = True


def kbScores(kb, correct_annotations, incorrect_annotations, keyword1, keyword2):
    correct_scores = []
    for a in correct_annotations:
        for x in a:

            time.sleep(1)
            score = kb.get_annotation_score(x, keyword1)
            correct_scores.append(score)

    incorrect_scores = []
    for a in incorrect_annotations:
        for x in a:
            time.sleep(1)
            score = kb.get_annotation_score(x, keyword2)
            incorrect_scores.append(score)

    return correct_scores, incorrect_scores


def calculatePerplexity(previous_annotations, new_annotation):
    perp = perplexity.Perplexity(previous_annotations)

    p_score = perp.score(new_annotation)

    return p_score


def testPerplexity(image_class):
    d = dataset.Dataset()
    dictionary = d.get_all_data()
    correct_annotations = d.additional_annotations

    p_correct_scores = []
    p_incorrect_scores = []
    count = 0
    for key in correct_annotations:
        count = count + 1
        a = getAnnotations(dictionary, (image_class, key))
        previous_annotations = a[0]
        incorrect_annotations = a[1]
        p_correct_scores.append(calculatePerplexity(previous_annotations, correct_annotations.get(key)))
        p_incorrect_scores.append(calculatePerplexity(previous_annotations, incorrect_annotations[count]))

    graph(p_correct_scores, p_incorrect_scores, 'perplexity.png')
    return p_correct_scores, p_incorrect_scores


def main():

    image1 = ("trains", 22)
    image2 = ("computers", 15)
    setUp(image1[0], image2[0])
    showImage(107943)
    c1 = classifier.Classifier()
    c1.loadClassifier()
    giveDemo(c1, image1)

    print(testPerplexity(image1[0]))




if __name__ == "__main__":
    main()


# current image
# showImage(22)
# interesting images
# showImage(107943)
# showImage(150311)
