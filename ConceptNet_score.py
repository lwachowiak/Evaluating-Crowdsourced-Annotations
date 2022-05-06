import requests
import stanza

# return relatedness score between two concepts in ConceptNet
def get_concept_relatedness_score(c1,c2):
    obj = requests.get("https://api.conceptnet.io/relatedness?node1=/c/en/"+c1+"&node2=/c/en/"+c2).json()
    return obj["value"]

# extract nouns from the annotation
def extract_nouns(annotation):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
    doc = nlp(annotation)
    list_of_nouns=[]
    for sent in doc.sentences:
        for w in sent.words:
            if w.upos in ["NOUN", "PROPN"]:
                noun=w.text
            # add additional word if compound 
                if w.deprel=="compound":
                    for w2 in sent.words:
                        if w2.id==w.head:
                            noun+=" "+w2.text
                list_of_nouns.append(noun)
    return list_of_nouns



def get_annotation_score(new_annotation, keyword):
    # extract the concepts from the new annotation
    concepts = extract_nouns(new_annotation)
    scores=[]
    for concept in concepts:
        scores.append(get_concept_relatedness_score(concept, keyword))
    avg_score=sum(scores)/len(scores)
    # print concepts and scores side by side
    for i in range(len(concepts)):  
        print(concepts[i], scores[i])
    return avg_score

print("Average Score:",get_annotation_score("the picture contains a cute puppy", "dog"))