import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# get list of all speech files
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])

# Output files
print(files)

# read each speech file
speeches = [read_file(document) for document in files]


# preprocess each speech
processed_speeches = process_speeches(speeches)


# merge speeches
all_sentences = merge_speeches(processed_speeches)


# view most frequently used words
most_freq_words = most_frequent_words(all_sentences)

# Print out most_freq_words
#print(most_freq_words)

# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom = all_prez_embeddings.most_similar("freedom", topn=20)

# Output similar_to_freedom
print("\n{}".format(similar_to_freedom))

# view words similar to unity
similar_to_unity = all_prez_embeddings.most_similar("unity", topn=20)

# Output similar_to_unity
print("\n{}\n".format(similar_to_unity))

# get President Roosevelt sentences
roosevelt_sentences = get_president_sentences("franklin-d-roosevelt")


# view most frequently used words of Roosevelt
roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)

# Output roosevelt_most_freq_words
#print(roosevelt_most_freq_words)


# create gensim model for Roosevelt
roosevelt_embeddings = gensim.models.Word2Vec(roosevelt_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for Roosevelt
roosevelt_similar_to_freedom = roosevelt_embeddings.most_similar("freedom", topn=20)

# Output roosevelt_embeddings
print("\n{}\n".format(roosevelt_similar_to_freedom))

# get sentences of multiple presidents
rushmore_prez_sentences = get_presidents_sentences(["washington","jefferson","lincoln","theodore-roosevelt"])

# View rushmore_prez_sentences
print(rushmore_prez_sentences)

# view most frequently used words of presidents
rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)

# Display a new line
print("\n")

# Print out rushmore_most_freq_words
#print(rushmore_most_freq_words)


# create gensim model for the presidents
rushmore_embeddings = gensim.models.Word2Vec(rushmore_prez_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for presidents
rushmore_similar_to_freedom = rushmore_embeddings.most_similar("freedom", topn=20)

# View rushmore_similar_to_freedom
#print(rushmore_similar_to_freedom)

# view words similar to devotion for presidents
rushmore_similar_to_devotion = rushmore_embeddings.most_similar("devotion", topn=20)

# View rushmore_similar_to_devotion
print(rushmore_similar_to_devotion)



# Function to accept president's name and return words similar to freedom from their speeches
def sentences_model(president_name):
  # Get President's sentences
  president_sentences = get_president_sentences(president_name)
  # Get the most frequently used words of the president
  freq_word = most_frequent_words(president_sentences)
  # Create gensim model for the president
  model_embeddings = gensim.models.Word2Vec(president_sentences, size=96, window=5, min_count=1, workers=2, sg=1)
  # Get words similar to freedom for the president
  words_similar_to_freedom = model_embeddings.most_similar("freedom", topn=20)
  # Return similar words
  return words_similar_to_freedom

# Display a new line
print("\n")

# View words similar to freedom for William Harrison
print(sentences_model("William-Harrison"))

# Display a new line
print("\n")

# View words similar to freedom for John Q Adams
print(sentences_model("John-Q-Adams"))