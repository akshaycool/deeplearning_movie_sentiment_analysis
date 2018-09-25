import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import nltk
import sys
import logging
import gensim
from gensim.models import word2vec,Word2Vec
from sklearn.ensemble import RandomForestClassifier
from word2vec_utility import Word2VecUtility
import os

reload(sys)
#To fix the unicode issues set the default encoding to the given below (nltk specific)
#http://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
sys.setdefaultencoding("ISO-8859-1")

#Function to split the review into words
def review_to_wordlist(review,remove_stopwords=False):
    #1. remove html tags
    review_text = BeautifulSoup(review).get_text()
    #2. remove non letters
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    #3. lower the text and split into words
    words = review_text.lower().split()

    #optionally remove the stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    #Return the list of words (opposed to sentenence in the bag_of_words_model)
    return words


#Function to split the review to parsed sentences
def review_to_sentences(review,tokenizer,remove_stopwords=False):
    #Returns the list of sentences, where each sentence is a list of words

    #1. Use NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    #2. Loop over sentences and construct the review_sentences list
    sentences = []

    for raw_sentence in raw_sentences:
        # if empty skip the sentence
        if len(raw_sentence)>0:
            sentences.append(Word2VecUtility.review_to_wordlist(raw_sentence,remove_stopwords))

    return sentences

def makeFeatureVec(words,model,num_features):
    #Function to average all of the word vecotors in a given paragraph

    #Pre initialize an empty numpy array(for speed)
    featureVec = np.zeros((num_features,),dtype='float32')

    nwords = 0

    #in the update of gensim , we need to use model.wv to access params
    index2wordset = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2wordset:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)

    return featureVec

def getAvgFeatureVecs(reviews,model,num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

if __name__ == '__main__':


    #load labeled training data 
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    
    #load test data
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )

    #load unlabeled training data
    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data','unlabeledTrainData.tsv',header=0,delimiter="\t",quoting=3))

    #downloads the punkt tokenizer from nltk
    nltk.download('punkt')

    #Load the punekt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []

    print "Parsing sentences from the training set"
    for review in train["review"]:
        sentences+=review_to_sentences(review,tokenizer)

    print "Parsing sentences from the unlabeled training set"
    for review in unlabeled_train["review"]:
        sentences+=review_to_sentences(review,tokenizer)




    logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s',level=logging.INFO)

    #Set values for different parameters for the model
    #(experimentation possible)
    num_features = 300 #Word vector dimensionality
    min_word_count = 40 #Minimum word count
    num_workers = 4 #Number of threads to run in parallel
    context = 10 #Context window size
    downsampling = 1e-3 #Downsample setting for frequent words

    #Initialize and training the model
    print "Training model...."
    model = word2vec.Word2Vec(sentences,workers=num_workers,
        min_count=min_word_count,window=context,
        sample = downsampling,size = num_features)

    #This makes the model much more memory efficient
    model.init_sims(replace=True)

    #Helpful to create a meaningful name and save for later use
    model_name = "300features_40minwords_10context"
    model.save(model_name)


    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.doesnt_match("paris berlin london austria".split())
    model.most_similar("man")
    model.most_similar("queen")
    model.most_similar("awful")



    # ****************************************************************
    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop word
    # removal.
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append( Word2VecUtility.review_to_wordlist( review, \
            remove_stopwords=True ))

    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features )

    print "Creating average feature vecs for test reviews"
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append( Word2VecUtility.review_to_wordlist( review, \
            remove_stopwords=True ))

    testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

    forest = RandomForestClassifier(n_estimators = 100)

    print "Fitting a random forest to labeled training set"
    forest = forest.fit(trainDataVecs,train["sentiment"])

    # Test & extract results
    result = forest.predict(testDataVecs)

    #Write the results to csv
    output = pd.DataFrame({"id":test['id'],"sentiment":result})
    output.to_csv("Word2Vec_AverageVectors.csv",index=False,quoting=3)
    print "Wrote Word2Vec_AverageVectors.csv"
