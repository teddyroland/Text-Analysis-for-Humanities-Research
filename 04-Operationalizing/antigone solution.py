import pandas
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


## Read in the text of Antigone

with open('antigone.txt', 'r') as file_in:
    antigone_text = file_in.read()


## Collect the dialogue spoken by each character

antigone_list = antigone_text.split('\n\n')
dialogue_dict = {}
for line in antigone_list:
    index_first_space = line.index(' ')
    character_name = line[:index_first_space]
    if character_name not in dialogue_dict.keys():
        dialogue_dict[character_name] = line[index_first_space:]
    else:
        dialogue_dict[character_name] = dialogue_dict[character_name] + line[index_first_space:]

dialogue_list = dialogue_dict.values()
character_list = dialogue_dict.keys()


## Pre-Process the text

english_stop_words = stopwords.words('english')
ye_olde_stop_words = ['thou','thy','thee', 'ye', 'hath','hast', 'wilt',\
                      'art', 'dost','doth','shalt','tis','canst','thyself']
all_stop_words = english_stop_words + ye_olde_stop_words

cv = CountVectorizer(stop_words=all_stop_words)
dtm = cv.fit_transform(dialogue_list)
dtm_df = pandas.DataFrame(dtm.toarray(), columns = cv.get_feature_names(), index = character_list)



## Most Distinctive Words

# Create new, empty dataframe
mdw_df = pandas.DataFrame()

# Add a column for her observed word counts
mdw_df['ANTIGONE'] = dtm_df.loc['ANTIGONE']

# Add a column for the total counts of each word in the play
mdw_df['WORD_TOTAL'] = dtm_df.sum()

# Get the total number of words spoken in the play
total_counts = mdw_df['WORD_TOTAL'].sum()

# Calculate Antigone's share of the total dialogue
char_space = mdw_df['ANTIGONE'].sum()/total_counts

# Add a new column for an "expected" number of times
# Antigone would utter each word
mdw_df['ANTIGONE_EXPECTED'] = mdw_df['WORD_TOTAL']*char_space

# How much more/less frequently does Antigone use the word than expected?
mdw_df['OBS-EXP_RATIO'] = mdw_df['ANTIGONE']/mdw_df['ANTIGONE_EXPECTED']

# Pull out rows in which the ratio is greater than 1
high_ratio_df = mdw_df[mdw_df['OBS-EXP_RATIO']>1]

# Pull out rows in which Antigone uses the word more than 5 times
high_frequency_df = high_ratio_df[high_ratio_df['ANTIGONE']>5]

# Get a list of Antigone's most frequent words, sorted by O/E Ratio
high_frequency_df.sort_values('OBS-EXP_RATIO', ascending=False)
