import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
import re
from word2vec_utility import Word2VecUtility
import os

if __name__=='__main__':
	#getting the training data of sentiments
	'''
	header = 0 -> First line of the file contains column names
	quoting = 3 -> tells python to ignore the doubled quotes
	'''
	train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
	test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3)

	clean_train_reviews = []
	num_reviews = train["review"].size

	for i in xrange(0,num_reviews):
		clean_train_reviews.append(" ".join(Word2VecUtility.review_to_wordlist(train["review"][i],True)))

	print "Creating the bag of words... \n"
	vectorizer = CountVectorizer(analyzer="word",
		tokenizer=None,
		preprocessor = None,
		stop_words = None,
		max_features = 5000)

	#Learns the vocabulary and returns the term document matrix (s
	#type(train_data_features) -> scipy.sparse.csr.csr_matrix
	train_data_features = vectorizer.fit_transform(clean_train_reviews)

	#converting to numpy array
	train_data_features = train_data_features.toarray()

	# Get the words in the vocabulary
	vocab = vectorizer.get_feature_names()
	print vocab

	#Sum up the counts of each vocabulary
	dist = np.sum(train_data_features,axis=0)

	#Initializer a Random Forest Classifier with 100 trees
	forest =RandomForestClassifier(n_estimators = 100)

	#Fit the forest to the training set, using the bag of words as
	#features and sentiment labels as the response variable

	# This may take a few minutes to run
	forest = forest.fit(train_data_features,train["sentiment"])

	#Evaluating the test data set

	clean_test_reviews = []
	num_reviews = len(test["review"])

	for i in xrange(0,num_reviews):
		if((i+1) % 1000 == 0):
			print "Review %d of %d\n" % (i+1, num_reviews)
		clean_test_reviews.append(" ".join(Word2VecUtility.review_to_wordlist(test["review"][i],True)))

	# Get a bag of words for the test set, and convert to a numpy array
	test_data_features = vectorizer.transform(clean_test_reviews)
	test_data_features = test_data_features.toarray()

	# Use the random forest to make sentiment label predictions
	result = forest.predict(test_data_features)

	output = pd.DataFrame(data = {"id":test['id'],"sentiment":result})

	#Use pandas to write the comma seperated file
	output.to_csv("Bag_of_Words_Model.csv",index= False,quoting=3)
