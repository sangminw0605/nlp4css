from transformers import pipeline


def summarize(model_name, data=None):
    summarizer = pipeline("summarization", model=model_name)

    ARTICLE = """ Bill Kristol and Ben Shaprio, two turds in the same toilet bowl.
    """

    for i in range(5):
        print(summarizer(ARTICLE, do_sample=True))


if __name__ == "__main__":
    print(summarize("facebook/bart-large-cnn"))