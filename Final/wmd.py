import spacy
import wmd

nlp = spacy.load('en_core_web_md')
nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
doc1 = nlp("Politician speaks to the media in Illinois.")
doc2 = nlp("The president greets the press in Chicago.")
print(doc1.similarity(doc2))