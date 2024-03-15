import math
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from tqdm import tqdm

from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)
np.random.seed(42)


def get_data(Z_bias=0.05, U_bias=5.0, constant=-0.5):
    """Compile the data from the 20 News Group Scikit-learn distribution
    
    Create a simulated dataset based on a type of news, where the varia-
    ble U is a confounder (whether or not a news document belongs to the
    category or not), Z is the observed variable dependent on U, and Y 
    is the target variable dependent on Z and the confounder.

    Args:
        Z_bias (float, optional): bias of var Z on target. Defaults to 0.05.
        U_bias (float, optional): bias of confounder on target. Defaults to 5.0.
        constant (float, optional): constant bias. Defaults to -0.5.

    Returns:
        np.array: simulated data containing where columns correspond to [Y, Z, U]
        List[str]: list of text in dataset
        vocabulary: vocabulary extracted from text
    """
    newsgroup_data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), shuffle=False)

    target_name_to_label = {
    'comp.graphics': 1,
    'comp.os.ms-windows.misc': 1,
    'comp.sys.ibm.pc.hardware': 1,
    'comp.sys.mac.hardware': 1,
    'comp.windows.x': 1,
    'misc.forsale': 2,
    'rec.autos': 3,
    'rec.motorcycles': 3,
    'rec.sport.baseball': 4,
    'rec.sport.hockey': 4,
    'sci.crypt': 5,
    'sci.electronics': 6,
    'sci.med': 7,
    'sci.space': 8,
    'alt.atheism': 9,
    'soc.religion.christian': 9,
    'talk.religion.misc': 9, 
    'talk.politics.guns': 10,
    'talk.politics.mideast': 10,
    'talk.politics.misc': 10,
    }
    label_names = ["comp", "sale", "auto", "sport", "crypt", "electronics", "med", "space", "religion", "politics"]
    labels = [target_name_to_label[newsgroup_data.target_names[original_label]]
              for original_label in newsgroup_data.target]
    labels = np.array(labels)

    vectorizer = CountVectorizer(min_df=25, max_df=0.01)
    vectorizer.fit(newsgroup_data.data)
    
    dropped_document_idx = set([])
    vocabset = set(vectorizer.get_feature_names_out())
    documents = []
    for idx, text in enumerate(newsgroup_data.data):
        document = []
        for token in text.split(" "):
            if token not in vocabset:
                continue
            document.append(token)

        if len(document) == 0:
            dropped_document_idx.add(idx)
            continue
        documents.append(text)
    labels = np.array([labels[idx]
                    for idx in range(len(labels))
                    if idx not in dropped_document_idx])
    assert(labels.shape[0] == len(documents))
  

    simulated_data = []
    for idx in tqdm(range(len(documents))):
        # confounder
        U = int(labels[idx]==9) # unobserved confounder = religion topic, influences text

        # treatment
        Z = int(1.0*U + np.random.normal(loc=0, scale=1) > 0) # Z depends on U
        Y = constant + Z_bias*Z + U_bias*U + np.random.normal(0.0, 0.1)

        simulated_data.append([Y, Z, U])

    simulated_data = np.array(simulated_data)
    return simulated_data, documents, vectorizer.vocabulary_

def regress_y_on_z(data, maxiter=100):
    """Use the statsmodels library to estimate the treatment effect
    by regression using the sm.OLS model given the variable Z and Y

    Args:
        data (np.array): simulated data
        maxiter: max number of iterations for the regression
    Outputs:
        Use .summary to print results (no return value)
    """

    #format data
    #create OLS model
    model = sm.OLS(data[:, 0], sm.add_constant(data[:, 1]))
    
    res = model.fit(method='pinv', maxiter=maxiter)
    print(res.summary(yname='Y', xname=['const', 'Z' ]))
    

def regress_y_on_z_and_u(data, maxiter=2000):
    """Use the statsmodels library to estimate the treatment effect
    by regression using the sm.OLS model given the Z & U and Y

    Args:
        data (np.array): simulated data
        maxiter: max number of iterations for the regression
    Outputs:
        Use .summary to print results (no return value)
    """
    #format data
    #create OLS model
    model = sm.OLS(data[:, 0], sm.add_constant(np.column_stack((data[:,1], data[:,2]))))
    
    res = model.fit(method='pinv', maxiter=maxiter)
    print(res.summary(yname='Y', xname=['const', 'Z', 'U']))

def regress_y_on_z_and_topics(data, documents, vocabulary, num_topics=50, maxiter=200):
    """Use a topic model to represent text as a proxy for the unknown confounder
     Then use the statsmodels library to estimate the treatment effect
    by regression using the sm.OLS model given the Z & topics and Y

    Args:
        data (np.array): simulated data
        documents: text data as returned by get_data
        vocabulary: vocabulary as returned by get_data
        num_topics: number of topics for topic model
        maxiter: max number of iterations for the regression
    Outputs:
        Use .summary to print results (no return value)
    """
    pass
    

# Return the adjusted and unadjusted average treatment effect
def reweigh_with_propensity_scores(data, documents, vocabulary):
    """Given the simulated data, documents, and vocab, reweigh
       the data with inverse probability weighting


    Args:
        data (np.array): simulated data
        documents: text data as returned by get_data
        vocabulary: vocabulary as returned by get_data
    Outputs:
        return [unadjusted ATE], [adjusted ATE]
    """
    pass
    

if __name__ == "__main__":
    Z_bias = 0.05
    U_bias = 5.0
    constant = -0.5
    data, documents, vocab = get_data(Z_bias=Z_bias, U_bias=U_bias, constant=constant)
    print(f"DATA: Y = {constant} + {Z_bias}*Z + {U_bias}*U")
    
    print("\nEstimating the treatment effect by regressing Y on Z only\n")
    regress_y_on_z(data)
    
    print("\n*******************************************************************************************")
    print("Estimating the treatment effect by regressing Y on Z and the confounder U\n")
    regress_y_on_z_and_u(data)
    
    exit() 
    
    print("\n*******************************************************************************************")
    print("Estimating the treatment effect by regressing Y on Z and the structured text\n")
    regress_y_on_z_and_topics(data, documents=documents, vocabulary=vocab, num_topics=20)
    
    print("\n*******************************************************************************************")
    print("Estimating the treatment effect by weighting with propensity scores\n")
    unadjusted, adjusted = reweigh_with_propensity_scores(data, documents=documents, vocabulary=vocab)
    print("Adjusted", adjusted)
    print("Unadjusted", unadjusted)
    
    