from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
import csv
from tqdm import tqdm
import ast

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def score(fname):
    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        currScore = 0
        i = 0

        for row in tqdm(spamreader):
            reference = ast.literal_eval(row[1])
            reference = [n.strip() for n in reference]
            sentences = [row[0]] + reference


            embeddings = model.encode(sentences)
            for s in embeddings[1:]:
                currScore += dot(embeddings[0], s)/(norm(embeddings[0])*norm(s))
                i += 1

    print(currScore / i)



if __name__ == "__main__":
    print('bart-large-cnn.csv:')
    score('bart-large-cnn.csv')

    print('bart-large-cnn-short.csv:')
    score('bart-large-cnn-short.csv')

    print('distilbart-cnn-12-6-short.csv:')
    score('distilbart-cnn-12-6-short.csv')

    print('gpt.csv:')
    score('gpt.csv')

    print('pegasus-xsum-short.csv')
    score('pegasus-xsum-short.csv')

    print('pegasus-xsum.csv')
    score('pegasus-xsum.csv')
