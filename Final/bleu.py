from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import csv
from tqdm import tqdm
import ast

rouge = evaluate.load('rouge')


def score(fname):
     
    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        iR = 0
        iB = 0
        bleuS = 0
        rougeS = 0

        for row in tqdm(spamreader):
            reference = ast.literal_eval(row[1])
            reference = [n.strip() for n in reference]
            candidate = row[0]

            for r in reference:
                results = rouge.compute(predictions=[candidate], references=[r])
                rougeS += results['rougeL']
                iR += 1

            reference = [r.split() for r in reference]

            chencherry = SmoothingFunction()

            # Calculate the BLEU Score
            bS = sentence_bleu(reference, candidate.split(), smoothing_function=chencherry.method1)

            if bS != 0:
                bleuS += bS
                iB += 1

    print('Rogue: ' + str(rougeS / iR))
    print('Bleu: ' + str(bleuS / iB))
    



if __name__ == "__main__":
    print('bart-large-cnn.csv:')
    score('bart-large-cnn.csv')

    print('bart-large-cnn-short.csv:')
    score('bart-large-cnn-short.csv')

    print('distilbart-cnn-12-6-short.csv:')
    score('distilbart-cnn-12-6-short.csv')

    print('distilbart-cnn-12-6.csv:')
    score('distilbart-cnn-12-6.csv')

    print('gpt.csv:')
    score('gpt.csv')

    print('pegasus-xsum-short.csv')
    score('pegasus-xsum-short.csv')

    print('pegasus-xsum.csv')
    score('pegasus-xsum.csv')

