from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ Bill Kristol and Ben Shaprio, two turds in the same toilet bowl.
"""

for i in range(5):
    print(summarizer(ARTICLE, do_sample=True))