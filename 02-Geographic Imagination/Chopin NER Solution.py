# Preparation
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from collections import Counter

modules = ["averaged_perceptron_tagger", "maxent_ne_chunker", "punkt"]

for module in modules:
    nltk.download(module)

    
# Read in text
with open('Chopin - The Awakening & Selected Short Stories.txt') as file_in:
    chopin_text = file_in.read()


# Named Entity Recognition
chopin_sents = sent_tokenize(chopin_text)
chopin_words = [word_tokenize(sent) for sent in chopin_sents]
chopin_pos = [pos_tag(sent) for sent in chopin_words]
chopin_ner = [ne_chunk(sent) for sent in chopin_pos]
chopin_chunks = [chunk for sent in chopin_ner for chunk in sent]
chopin_persons = [chunk.leaves() for chunk in chopin_chunks if type(chunk)==nltk.tree.Tree and chunk.label()=='GPE']
chopin_names_only = [name for person in chopin_persons for name,tag in person]
chopin_counted = Counter(chopin_names_only)
print(chopin_counted.most_common())
