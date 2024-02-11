import pandas
from collections import Counter, defaultdict
import math
from tabulate import tabulate

from helper import read_data, politics_words


def compute_log_odds_ratios(a_counter, b_counter):
    """ Calculate the log odds ratio given the counters for group A and
    group B.

    Parameters:
    ===========
    a_counter (collections.Counter): word counts for group A
    b_counter (collections.Counter): word counts for group B

    Returns:
    ========
    log_odds_ratio (dict): dictionary containing the log odds ratio of each
    word {"word" (str): ratio (float)}
    """ 
    log_odds_ratio = {}
    ####### PART A #######
    
    a_total = a_counter.total()
    b_total = b_counter.total()

    for word in a_counter:
        if word in b_counter:
            a_freq = a_counter[word]
            b_freq = b_counter[word]
            log_odds_ratio[word] = math.log(a_freq / (a_total - a_freq)) - math.log(b_freq / (b_total - b_freq))

    return log_odds_ratio


def compute_odds_with_prior(counts1, counts2, prior):
    """ Calculate the log odds ratio with a prior given the counters
    for group A, group B and the prior.

    Parameters:
    ===========
    a_counter (collections.Counter): word counts for group A
    b_counter (collections.Counter): word counts for group B
    prior     (collections.Counter): word counts for the prior
    
    Returns:
    ========
    log_odds_ratio (dict): dictionary containing the log odds ratio of each
    word {"word" (str): ratio (float)}
    """ 
    log_odds_ratio = {}

    ####### PART B #######

    a_total = counts1.total()
    b_total = counts2.total()
    p_total = prior.total()

    words = set(counts1) | set(counts2) 

    for word in words:
        a_count = counts1[word]
        b_count = counts2[word]
        p_count = prior[word]

        # omega computation
        a_omega = (a_count + p_count) / (a_total + p_total - a_count - p_count)
        b_omega = (b_count + p_count) / (b_total + p_total - b_count - p_count)

        # delta computation
        delta = math.log(a_omega) - math.log(b_omega)

        # sigma computation
        sigma = (1 / (a_count + p_count)) + (1 / (b_count + p_count))

        # zeta computation
        log_odds_ratio[word] = delta / math.sqrt(sigma)

    return log_odds_ratio



def smoother(A, window, count_list):
    """ smooth the counts given a window and a smoothing factor A.

    Parameters:
    ===========
    A            (float): smoothing factor
    window       (int): window length of the moving average
    count_list   (List[collections.Counter]): a list of counters to smooth
    
    Returns:
    ========
    new_counts    (List[collections.Counter]): a smoothed list of counters
    """ 
    new_counts = []

    ####### PART C #######
    
    prev = count_list[window - 1]
    for i in range(window, len(count_list)):
        smoothed = Counter()
        words = set()
        
        for j in range(i - window, i):
            words = words | set(count_list[j])
        
        for word in words:
            total = 0
            for j in range(i - window, i):
                total += count_list[j][word]
                    
            smoothed[word] = A * total + (1 - A) * prev[word]
            
        prev = smoothed
        new_counts.append(smoothed)

    return new_counts
            


if __name__ == "__main__":
    data_df = read_data()
    
    # Separate data
    r_df = data_df[data_df.party == "R"]
    d_df = data_df[data_df.party == "D"]

    r_counter = Counter()
    r_df.text.apply(lambda t: r_counter.update(t.split()))

    d_counter = Counter()
    d_df.text.apply(lambda t: d_counter.update(t.split()))
    
    # Part A. Compute Log-Odds Ratio
    w_to_ratio = compute_log_odds_ratios(d_counter, r_counter)
    print("More Republican")
    table = []
    for w, odds in sorted(w_to_ratio.items(), key=lambda item: item[1])[:10]:
        table.append([w, odds, r_counter[w], d_counter[w]])
    print(tabulate(table, headers=["Word", "Odds", "R count", "D count"]))

    print()
    print("More Democrat")
    table = []
    for w, odds in sorted(w_to_ratio.items(), key=lambda item: item[1], reverse=True)[:10]:
        table.append([w, odds, r_counter[w], d_counter[w]])
    print(tabulate(table, headers=["Word", "Odds", "R count", "D count"]))
    
    # Part B. Compute Log-Odds Ratio with Prior
    prior = d_counter
    prior.update(r_counter)

    w_to_ratio = compute_odds_with_prior(d_counter, r_counter, prior)
    print()
    print("More Republican")
    table = []
    for w, odds in sorted(w_to_ratio.items(), key=lambda item: item[1])[:10]:
        table.append([w, odds, r_counter[w], d_counter[w]])
    print(tabulate(table, headers=["Word", "Odds", "R count", "D count"]))


    print()
    print("More Democrat")
    table = []
    for w, odds in sorted(w_to_ratio.items(), key=lambda item: item[1], reverse=True)[:10]:
       table.append([w, odds, r_counter[w], d_counter[w]])
    print(tabulate(table, headers=["Word", "Odds", "R count", "D count"]))

        
    # Part C. Compute word evolutions
    d_counters = []
    r_counters = []
    
    for i, session_id in enumerate(data_df.session_id.unique()):
        r_df = data_df[(data_df.party == "R")&(data_df.session_id == session_id)]
        d_df = data_df[(data_df.party == "D")&(data_df.session_id == session_id)]
        r_counter = Counter()
        r_df.text.apply(lambda t: r_counter.update(t.split()))
        r_counters.append(r_counter)

        d_counter = Counter()
        d_df.text.apply(lambda t: d_counter.update(t.split()))
        d_counters.append(d_counter)
    new_d_counters = smoother(A=.2, window=1, count_list=d_counters)
    new_r_counters = smoother(A=.2, window=1, count_list=r_counters)
    
    print()
    print("Changes over time in log odds with prior")
    political_keywords = {w: [] for w in politics_words}
    for i, (d_counter, r_counter) in enumerate(zip(new_d_counters, new_r_counters)):
        w_to_ratio = compute_odds_with_prior(d_counter, r_counter, prior)
        for w, odds_list in political_keywords.items():
            political_keywords[w].append(w_to_ratio[w])
    table = []
    for w, odds_list in political_keywords.items():
        table.append([w] + odds_list)
    print(tabulate(table, headers=["Word", "112", "113", "114"]))