from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate

rouge = evaluate.load('rouge')

# Example of reference translations and machine translation
reference = ["jewish people lied about the holocaust", "jews are liars", "jews didn't die in the holocaust", "trivializes harm to victims"]
candidate = "Minimizing Jewish suffering in holocaust."

for r in reference:
    results = rouge.compute(predictions=[candidate], references=[r])
    print(results)

reference = [r.split() for r in reference]

chencherry = SmoothingFunction()

# Calculate the BLEU Score
print('BLEU score -> {}'.format(sentence_bleu(reference, candidate.split(), smoothing_function=chencherry.method1)))