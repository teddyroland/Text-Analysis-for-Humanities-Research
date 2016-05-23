import os
from collections import Counter
from nltk import word_tokenize, NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


### Prepare NLTK resources

stopword_set = set(stopwords.words('english'))
wnl = WordNetLemmatizer()


### Import, tokenize, lemmatize movie reviews

negative_path = 'movie_reviews/negative/'
negative_files = os.listdir(negative_path)
negative_volumes = [open(negative_path+name,'r').read() for name in negative_files]
negative_tokenized = [word_tokenize(review.lower()) for review in negative_volumes]
negative_no_stops = [[word for word in volume if word not in stopword_set] for volume in negative_tokenized]
negative_lemmatized = [[wnl.lemmatize(word) for word in review] for review in negative_no_stops]
negative_sets = [set(volume) for volume in negative_lemmatized]

positive_path = 'movie_reviews/positive/'
positive_files = os.listdir(positive_path)
positive_volumes = [open(positive_path+name,'r').read() for name in positive_files]
positive_tokenized = [word_tokenize(review.lower()) for review in positive_volumes]
positive_no_stops = [[word for word in review if word not in stopword_set] for review in positive_tokenized]
positive_lemmatized = [[wnl.lemmatize(word) for word in review] for review in positive_no_stops]
positive_sets = [set(review) for review in positive_lemmatized]


### Create a list of most common words across entire corpus, remove uncommon words

all_sets = negative_sets + positive_sets
term_document_frequency_list = [word for review in all_sets for word in review]
tdf_counts = Counter(term_document_frequency_list)
more_than_once = [key for key in tdf_counts.keys() if tdf_counts[key]>1]
more_than_once_set = set(more_than_once)

negative_min_df = [[word for word in review if word in more_than_once_set] for review in negative_sets]
positive_min_df = [[word for word in review if word in more_than_once_set] for review in positive_sets]


### Get our texts into the format NLTK expects for its classifier

negative_featurized = [{word:True for word in review} for review in negative_min_df]
positive_featurized = [{word:True for word in review} for review in positive_min_df]

negative_tagged = [(review,'negative') for review in negative_featurized]
positive_tagged = [(review,'positive') for review in positive_featurized]

all_tagged = negative_tagged + positive_tagged


### Train the classifier

classifier = NaiveBayesClassifier.train(all_tagged)


### Import, process, featurize new set of movie reviews

ebert_path = 'movie_reviews/ebert/'
ebert_files = os.listdir(ebert_path)
ebert_reviews = [open(ebert_path+name).read() for name in ebert_files]
ebert_tokenized = [word_tokenize(review.lower()) for review in ebert_reviews]
ebert_no_stops = [[word for word in review if word not in stopword_set] for review in ebert_tokenized]
ebert_lemmatized = [[wnl.lemmatize(word) for word in review] for review in ebert_tokenized]
ebert_set = [set(review) for review in ebert_lemmatized]
ebert_min_df = [[word for word in review if word in more_than_once_set] for review in ebert_set]
ebert_featurized = ({word:True for word in review} for review in ebert_min_df)


print(classifier.classify_many(ebert_featurized))

