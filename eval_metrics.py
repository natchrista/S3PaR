# tutorial 1: https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54
# tutorial 2: https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832
# tutorial 3: https://towardsdatascience.com/20-popular-machine-learning-metrics-part-2-ranking-statistical-metrics-22c3e5a937b6

import numpy as np

def dcg_at_k(r, k, method = 0):
    '''
    references:
    > https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1
    > https://www.kaggle.com/code/wendykan/ndcg-example

    Function to calculate Discounted Cummulative Gain
    Cummulative gain is the sum of all the relevance scores in a recommendation set.
    Discounted Cummulative Gain is the sum of relevance score divided by the log of the corresponding position

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider -- the K value in topK
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    '''
    r = np.asarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')

def ndcg_at_k(r, k, method = 0):
    '''
    references:
    > https://towardsdatascience.com/evaluate-your-recommendation-engine-using-ndcg-759a851452d1
    > https://www.kaggle.com/code/wendykan/ndcg-example

    Function to calculate NORMALIZED Discounted Cummulative Gain

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    '''
    dcg_max = dcg_at_k(sorted(r, reverse = True), k, method) # this is the ideal order of relevancy, i.e., highest relevant result first, thus the sorting
    if not dcg_max:
        return 0
    else:
        dcg_k = dcg_at_k(r, k, method)
        returned = dcg_k / dcg_max
        return returned

def prc(prediction, gold):
    '''
    Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    note: there is only one label for gold --> check if the gold exist in the prediction, if not then return 0/
    '''
    prediction_int = [int(item) for item in prediction]
    relevant_item = gold

    if relevant_item in prediction_int:
        return float(1.0/len(prediction))
    else:
        return int(0)

def rec(prediction, gold):
    '''
    Recall@k = (Relevant_Items_Recommended in top-k) / (Relevant_Items)
    '''
    prediction_int = [int(item) for item in prediction]
    relevant_item = gold # relevant item is always 1 in my task
    if relevant_item in prediction_int:
        return int(1)
    else:
        return int(0)

def mrr(prediction, gold):
    '''
    '''
    prediction_int = [int(item) for item in prediction]
    for i in range(len(prediction_int)):
        if prediction_int[i] == gold:
            return float(1.0/(i+1.0))
        else:
            return int(0)

def acc(prediction, gold):
    prediction_int = [int(item) for item in prediction]
    if gold in prediction_int:
        return int(1)
    else:
        return int(0)

def avg(input_list, total_instances):
    '''
    For final averaging after collecting all scores per top-K
    '''
    return float(sum(input_list)/total_instances)

def hamming_score(y_true, y_pred):
    '''
    reference = https://www.linkedin.com/pulse/hamming-score-multi-label-classification-chandra-sharat
    '''
    acc_list = []

    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)

    return np.mean(acc_list)
