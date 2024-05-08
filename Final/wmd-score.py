import gensim.downloader as api

def preprocess(sentence):
    return [w for w in sentence.lower().split()]


sentence_obama = "Toxic masculinity: drinking beer, objectifying women."
sentence_president = "men need to prove themselves as manly"

sentence_obama = preprocess(sentence_obama)
sentence_president = preprocess(sentence_president)


model = api.load('word2vec-google-news-300')

distance = model.wmdistance(sentence_obama, sentence_president)

print('distance = %.4f' % distance)