from nltk.probability import FreqDist

with open('christ-and-satan.txt') as f:
    cs_text = f.read()

word_list = cs_text.split()
first_letter = [word[0] for word in word_list]
letter_dist = FreqDist(first_letter)
letter_dist.plot(4,cumulative=True)
