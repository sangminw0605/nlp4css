from transformers import pipeline
import csv
from tqdm import tqdm

def summarize(model_name, data=None):
    summarizer = pipeline("summarization", model=model_name)

    with open(data, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        with open(model_name.split('/')[-1] + '.csv', 'w', newline='') as newfile:
            spamwriter = csv.writer(newfile, delimiter=',', quotechar='"')
            for row in tqdm(spamreader):
                summary = summarizer(row[0], do_sample=True)
                spamwriter.writerow([summary[0]['summary_text']])


if __name__ == "__main__":
    summarize("facebook/bart-large-cnn", data="data.csv")