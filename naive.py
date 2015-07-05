import pickledb
import nltk
import  math
import  re

db = pickledb.load('bayes.txt', False)


def addClass(label):
    classes = getClasses()
    if label not in classes:
        classes.append(label)
        db.set('bayes_classes', classes)
    return True


def getClasses():
    classes = db.get('bayes_classes')
    if classes is None:
        classes = []
    return classes


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens


def incrementWordCount(word, classLabel):
    update('bayes_wordCount::' + word)
    update('bayes_word_' + word + '_class_' + classLabel)


def update(key):
    count = db.get(key)
    if count is None:
        count = 0
    db.set(key, count + 1)
    return count + 1


def getInverseWordClassCount(word,label):
    classes = getClasses()
    inverseCount = 0
    for i in classes:
        if i == label:
            continue
        inverseCount = inverseCount + getwordclasscount(word,i)
    return inverseCount

def getwordclasscount(word,label):
    count = db.get('bayes_word_' + word + '_class_' + label)
    if count is None:
        return 0
    return count

def getwordcount(word):
    count = db.get('bayes_wordCount::' + word)
    if count is None:
        return 0
    return count


def incrementTotalCount(label):
    return update('bayes_total_count_' + label)


def class_dictionary_count(classLabel):
    count = db.get('bayes_total_count_' + classLabel)
    return count






def bayes_learn(text, label):
    addClass(label)
    words = tokenize(text)
    length = len(words)
    for x in range(0, length):
        incrementWordCount(words[x], label)
    incrementTotalCount(label)

def bayes_predict(text):
    words = tokenize(text)
    length = len(words)
    classes = getClasses()
    class_dictionary = {}
    iclass_dictionary = {}
    scores = {}
    classProbability = {}
    totalCount = 0
    for x in range(0,len(classes)):
        thisClass = classes[x]
        class_dictionary[thisClass] = class_dictionary_count(thisClass)
        totalCount = totalCount + class_dictionary[thisClass]

    for j in range(0,len(classes)):
        thisClass = classes[j]
        iclass_dictionary[thisClass] = totalCount - class_dictionary[thisClass]

    for y in range(0,len(classes)):
        thisClass = classes[y]
        sum = 0
        classProbability[thisClass] = class_dictionary[thisClass] / totalCount
        for word in words:
            wordCount = getwordcount(word)
            if wordCount == 0:
                continue
            else:
                wordProbability =  getwordclasscount(word,thisClass) / class_dictionary[thisClass]
                wordInverseProbability  = getInverseWordClassCount(word,thisClass) / iclass_dictionary[thisClass]
                word_conditional_probability = wordProbability / (wordProbability + wordInverseProbability)
                word_conditional_probability = (( 1 * 0.5 ) + ( wordCount * word_conditional_probability )) / ( 1 + wordCount)
                if word_conditional_probability == 0:
                    word_conditional_probability = 0.01
                elif word_conditional_probability == 1:
                    word_conditional_probability = 0.99

            sum = sum + ( math.log(1 - word_conditional_probability) - math.log(word_conditional_probability))
        scores[thisClass] = 1 / ( 1 + math.exp(sum))
    return  scores



def get_winner(scores):
    bestscore = 0
    bestclass = None
    for classes in scores:
        if scores[classes] > bestscore:
            bestscore = scores[classes]
            bestclass = classes
    print("Class : "+bestclass + " :: Score : "+str(bestscore*100)+"%") 



def readtrainingdata():
    with open("dataset") as f:
        for line in f:
            text = re.split(r'\t+',line)[1].rstrip()
            classLabel = re.split(r'\t+',line)[0]
            bayes_learn(text,classLabel)
    print("Train complete.. Predict now")
    predict()

def predict():
    scores = bayes_predict(input())
    get_winner(scores)


def start():
    print("train or predict : 1 or 2")
    answer  = int(input())
    if answer == 1:
        readtrainingdata()
    else:
        predict()



start()
db.dump()

