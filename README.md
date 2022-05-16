# Evaluating-Crowdsourced-Annotations

## Project

This Github repository contains the accompanying code to the paper titled Towards *Trustworthy Crowdsourced Annotations of Images in The National Archives*. Team consisting of L Thorburn, P Tisnikar, L Wachowiak, and M Waller developed a pipeline measuring the surprisal of a submitted annotation based on the context of the image class. For example, for an image that belongs to class "trains", the annotation "the mousepad is blue" might be more surprising, and possibly inappropriate. However, this annotation would not be surprising for an image of class "computers".

This metric captures this surprisal by 2 scores:
- Semantic similarity score, which is measured as a distance in the sentence embedding space of a language model (we use MiniLM-L6-v2)
- Conceptual relatedness score, which is measured as a distance in a knowledge graph (we use ConceptNet)

The two scores are computed per class, using all annotations within that class to produce an SVM classifier which then flags new annotations that are potentially irrelevant to the image.

The dataset we used for demonstration is [Visual Genome](https://visualgenome.org/), with a subset of images and custom annotations (available in ```dataset.py```).

## Repository

To install, clone the repository and then install the dependencies from the requirements.txt file using pip.

```
git clone https://github.com/lwachowiak/Evaluating-Crowdsourced-Annotations.git
pip install -r requirements.txt
```

To run the script, simply run the ```main.py``` within the ```SurpriseMetricMain``` folder.

The repository contains the following items:

* ```classifier.py``` contains the SVM classifier
* ```closeness.py``` contains the closeness metric calculation
* ```dataset.py``` contains the dataset constructors and custom annotations
* ```demo.py``` contains the demo routine
* ```kbscores.py``` contains the ConceptNet closeness score calculation
* ```main.py``` contains the main routine
* ```perplexity.py``` contains the perplexity calculation (not included in the classifier)

## Contact
name.surname@kcl.ac.uk
