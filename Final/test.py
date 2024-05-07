from transformers import pipeline
import csv
from tqdm import tqdm

def summarize(model_name, data=None, useParams=False):
    summarizer = pipeline("summarization", model=model_name)   
    summary = None
     

    with open(data, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        parameters = ""
        if useParams:
            parameters = '-short'

        with open(model_name.split('/')[-1] + parameters + '.csv', 'w', newline='') as newfile:
            spamwriter = csv.writer(newfile, delimiter=',', quotechar='"')
            for row in tqdm(spamreader):
                if useParams:
                    summary = summarizer(row[0], max_length=len(row[0].split()), min_length=0, do_sample=True)
                else:
                    summary = summarizer(row[0], do_sample=True)

                spamwriter.writerow([summary[0]['summary_text']] + [row[-1]])


if __name__ == "__main__":
    summarize("google/pegasus-xsum", data="data.csv", useParams=True)
    summarize("facebook/bart-large-cnn", data="data.csv", useParams=True)
    summarize("sshleifer/distilbart-cnn-12-6", data="data.csv", useParams=True)

    summarize("google/pegasus-xsum", data="data.csv")
    summarize("facebook/bart-large-cnn", data="data.csv")
    summarize("sshleifer/distilbart-cnn-12-6", data="data.csv")


