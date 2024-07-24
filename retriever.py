import os, json, csv
from tqdm import tqdm
from p_tqdm import p_map
import multiprocessing
from multiprocessing import Pool
from copy import deepcopy
from sentence_transformers import SentenceTransformer # for SBERT
import numpy as np
import pandas as pd
from scipy import spatial
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords
import math, ast, random
from math import *
from decimal import Decimal
from operator import itemgetter
from time import time
from eval_metrics import ndcg_at_k
from itertools import combinations
from sklearn.neighbors import KDTree
from utils import OntologyBuilder, DiversityCacheBuilder 
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class Retriever():
    def __init__(self,
                 predicted_clusters, 
                 input_papers, # format is {test_year : [input_papers]}
                 y_true_multilabel, # format is {test_year : [gold labels]}
                 papers_wrt_cluster, # format is {test_year: {cluster : [list of papers within that cluster]}}
                 result_folder_name, # current result's subfolder, usually the name is sth like ep4_iomode42_b32_lr00001_20220728113850036513
                 args,
                 last_n_papers = 3, # last n papers (n most recent papers) from the input sequence, default is set to 3
                 test_years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], # list of test years
                 all_k = [1, 5, 10, 15, 20, 30, 40, 50, 60],
                 threshold = 0.75,
                 relevance_range = 2, # 2 for binary 0 or 1 for relevant or not relevant, and 4 for 4 range relevance
                 use_learn_to_rank = False, # use learning to rank after reranking based on similarity, the boolean value can be obtained from args.ltr_in_sim (--use_ltr_in_sim in terminal)
                 ltr_model = None, # ranknet or lambdarank MODEL (i.e., the model has to be defined in the main code that calls Retriever class. For example, right now in temp_retrieval, i have to first call the RankNet or LambdaRank model)
                 paper_keywords = None,
                 paper_abstract = None,
                 input_authors = None,
                 use_similarity_based = True,
                 use_knn = False,
                 use_kdtree = False,
                 show_recommendation_samples = False,
                 compare_keywords_with_onto = False,
                 diversity_focus = False,
                 ):

        self.use_similarity_based = use_similarity_based
        self.use_knn = use_knn
        self.use_kdtree = use_kdtree
        if self.use_similarity_based:
            if self.use_knn or self.use_kdtree: # nyala semuanya
                print('MUST CHOOSE EITHER SIMILARITY-BASED OR KN OR KDTREE. Exiting... \n')
                exit()

        self.predicted_clusters = predicted_clusters
        self.input_papers = input_papers
        self.y_true_multilabel = y_true_multilabel
        self.papers_wrt_cluster = papers_wrt_cluster
        self.folder_name = 'run_results/{}/'.format(result_folder_name)
        self.test_years = test_years
        self.all_k = all_k
        self.threshold = threshold
        self.relevance_range = relevance_range
        self.use_learn_to_rank = use_learn_to_rank
        self.ltr_model = ltr_model
        self.args = args
        self.paper_encoder = json.load(open('../kg_links/all_modes_paper_encoder.json'))
        self.paper_decoder = {}
        self.recommender_samples = show_recommendation_samples
        if input_authors != None:
            self.recommender_samples = True,
            self.input_authors = input_authors
            self.author_papers = json.load(open('../data/author_to_year_to_papers.json'))
            self.paper_to_title = json.load(open('../data/paper_to_title.json'))
            self.paper_to_abstract = json.load(open('../data/paper_to_abstract.json'))
            self.paper_to_topics = json.load(open('../data/paper_to_topics.json'))

        for k, v in self.paper_encoder.items():
            self.paper_decoder[v] = k
        if paper_keywords == None and paper_abstract == None:
            self.paper_keywords = json.load(open('../data/paper_corresponding_topics_yake.json'))
            self.paper_abstract = json.load(open('../data/paper_paper_abstract_title_bert.json'))
        else:
            self.paper_keywords = paper_keywords
            self.paper_abstract = paper_abstract
        self.all_papers_in_data = set()

        print('Doing PAPER RETRIEVAL based on the results in folder {} with relevance range {}'.format(self.folder_name, self.relevance_range))

        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        try:
            print('Loading existing s_bert cache...')
            json_file =json.load(open('cache/sbert_cache.json'))
            self.sbert_cache = {}
            for k, v in json_file.items():
                list_v = [float(item) for item in v]
                self.sbert_cache[k] = np.array(list_v)
        except:
            self.sbert_cache = {}
        self.sim_cache = {}
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))

        self.avg_precision_dict = {}
        self.avg_ndcg_dict = {}

        for test_year in self.test_years:
            print('\nPROCESSING TEST YEAR {}\n'.format(test_year))
            output_samples = {}
            if self.args.recommendation_sample:
                curr_authors_temp = self.input_authors.get(test_year)
                curr_authors = []
                for item in curr_authors_temp:
                    curr_authors.append(item[0])

            curr_predictions = self.predicted_clusters.get(test_year)
            curr_inputs = self.input_papers.get(test_year)
            curr_gold_multilabels = self.y_true_multilabel.get(test_year)
            curr_year_papers_wrt_clusters = self.papers_wrt_cluster.get(test_year)
            for k, v in curr_year_papers_wrt_clusters.items():
                for ppr in v:
                    self.all_papers_in_data.add(ppr)
            assert len(curr_predictions) == len(curr_gold_multilabels)

            similar_papers = [None] * len(curr_predictions)
            top_k_precision = {}
            top_k_ndcg = {}
            top_k_recall = {}
            top_k_mrr = {}
            top_k_accuracy = {}
            top_k_diversity = {}

            for i in tqdm(range(len(curr_predictions))):
                curr_target_cluster = int(curr_predictions[i])
                curr_input_papers = curr_inputs[i][-last_n_papers:]
                curr_gold_papers = curr_gold_multilabels[i]
                papers_in_curr_cluster = curr_year_papers_wrt_clusters.get(curr_target_cluster)
                cluster_size = len(papers_in_curr_cluster)
                similarity_results = {}
                all_cluster_paper_embs = []

                if self.use_knn:
                    pass

                elif self.use_kdtree:
                    for cluster_paper in papers_in_curr_cluster:
                        cluster_ppr_emb = self.get_embedding(cluster_paper)
                        try:
                            emb_len = cluster_ppr_emb.shape[0]
                        except IndexError:
                            cluster_ppr_emb = np.zeros(384)
                        all_cluster_paper_embs.append(cluster_ppr_emb)

                    kdtree = KDTree(all_cluster_paper_embs, leaf_size = 6)

                    for input_paper in curr_input_papers:
                        input_emb = self.get_embedding(input_paper)
                        dist, ind = kdtree.query(np.array([input_emb]), k = cluster_size)
                        dist = dist.tolist()[0]
                        ind = ind.tolist()[0]
                        ind_and_dist = [(ind[i], dist[i]) for i in range(len(ind))]
                        for item in ind_and_dist:
                            indx, distance = item[0], item[-1]
                            cluster_paper = papers_in_curr_cluster[indx]
                            if cluster_paper not in similarity_results:
                                similarity_results[cluster_paper] = [distance]
                            else:
                                similarity_results[cluster_paper] += [distance]

                elif self.use_similarity_based:
                    for cluster_paper in papers_in_curr_cluster:
                        cluster_ppr_emb = self.get_embedding(cluster_paper)
                        for input_paper in curr_input_papers:
                            input_emb = self.get_embedding(input_paper)
                            similarity = self.find_sim_cache(cluster_paper, cluster_ppr_emb, input_paper, input_emb, allow_perfect_sim = False)
                            if cluster_paper not in similarity_results:
                                similarity_results[cluster_paper] = [similarity]
                            else:
                                similarity_results[cluster_paper] += [similarity]

                cluster_paper_and_optimal_sim = []
                for k, v in similarity_results.items():
                    if self.use_similarity_based:
                        max_sim = max(v)
                        paper_sim_tup = (k, max_sim)
                    elif self.use_kdtree:
                        min_sim = min(v)
                        paper_sim_tup = (k, min_sim)
                    cluster_paper_and_optimal_sim.append(paper_sim_tup)
                if self.use_similarity_based:
                    sorted_cluster_paper_and_optimal_sim = sorted(cluster_paper_and_optimal_sim, key = itemgetter(1), reverse = True)
                elif self.use_kdtree:
                    sorted_cluster_paper_and_optimal_sim = sorted(cluster_paper_and_optimal_sim, key = itemgetter(1), reverse = False)

                if self.use_learn_to_rank:
                    temp_sorted_cluster_paper = [item[0] for item in sorted_cluster_paper_and_optimal_sim] # list of paper ids
                    if self.ltr_model == None:
                        print('Since Learning to Rank is used, the Learning to Rank Model needs to be defined too! define either ranknet or lambdarank in the main code that calls this Retriever class (e.g., temp_retrieval.py)')
                        exit()
                    else:
                       ranking_model = self.ltr_model
                       ltr_sorted_index, sorted_cluster_paper = ranking_model.rerank(test_year, curr_input_papers, temp_sorted_cluster_paper)
                else:
                    sorted_cluster_paper = [item[0] for item in sorted_cluster_paper_and_optimal_sim]

                similar_papers[i] = sorted_cluster_paper

            assert len(similar_papers) == len(curr_gold_multilabels)

            different_len_counter = 0

            for i in tqdm(range(len(curr_gold_multilabels))):
                instance = i + 1
                curr_gold_papers = curr_gold_multilabels[i]
                curr_input_papers = curr_inputs[i][-last_n_papers:]
                curr_similar_papers = similar_papers[i]
                all_relevant_papers = set()

                for gold_paper in curr_gold_papers:
                    gold_emb = self.get_embedding(gold_paper)
                    for data_paper in self.all_papers_in_data:
                        data_emb = self.get_embedding(data_paper)
                        similarity = self.find_sim_cache(gold_paper, gold_emb, data_paper, data_emb)
                        if similarity >= self.threshold:
                            all_relevant_papers.add(data_paper)
                overall_relevant_items = len(all_relevant_papers)

                for k in all_k:
                    relevant_recommendations = set()
                    irrelevant_recommendations = set()
                    irrelevant_recommendations_indexes = set()
                    relevant_recommendations_indexes = set()
                    relevant_indexes = set()
                    relevance_scores = []

                    top_k_recommendations = curr_similar_papers[:k]

                    for l in range(len(top_k_recommendations)):
                        recommended = top_k_recommendations[l]
                        recommended_emb = self.get_embedding(recommended)
                        recommended_index = l + 1
                        curr_relevance_scores = []
                        for gold_paper in curr_gold_papers:
                            gold_emb = self.get_embedding(gold_paper)
                            similarity = self.find_sim_cache(recommended, recommended_emb, gold_paper, gold_emb)
                            if self.args.relevance_importance:
                                gold_paper_idx = curr_gold_papers.index(gold_paper) + 1
                                imp_score = 1/gold_paper_idx
                                if similarity >= self.threshold:
                                    final_score = similarity + imp_score
                                    relevant_recommendations.add(recommended)
                                    relevant_indexes.add(recommended_index)
                                    curr_relevance_scores.append(final_score)
                                elif similarity < self.threshold:
                                    curr_relevance_scores.append(0)
                            else:
                                if similarity >= self.threshold:
                                    relevant_recommendations.add(recommended)
                                    relevant_indexes.add(top_k_recommendations.index(recommended) + 1)
                                    curr_relevance_scores.append(1)
                                elif similarity < self.threshold:
                                    curr_relevance_scores.append(0)

                        try:
                            relevance_scores.append(max(curr_relevance_scores))
                        except Exception as e:
                            print(e)
                            pass

                    for l in range(len(top_k_recommendations)):
                        recommended = top_k_recommendations[l]
                        recommended_index = l + 1
                        if recommended not in relevant_recommendations:
                            irrelevant_recommendations.add(recommended)
                            irrelevant_recommendations_indexes.add(recommended_index)
                        else:
                            pass

                    relevant_counter = len(relevant_recommendations)
                    while len(relevance_scores) != k: 
                        relevance_scores.append(0)
                        different_len_counter += 1/(len(all_k)-1)

                    # calculating precision, recall, and MRR
                    ndcg = round(ndcg_at_k(relevance_scores, k, method = 1), 5)
                    precision = round(relevant_counter/int(k), 5)
                    if k not in top_k_precision:
                        top_k_precision[k] = [precision] 
                    else:
                        top_k_precision[k] += [precision]
                    if k not in top_k_ndcg:
                        top_k_ndcg[k] = [ndcg]
                    else:
                        top_k_ndcg[k] += [ndcg]

            print('Different len counter', different_len_counter)

            json_serializable_sbert_cache = {}
            for k, v in self.sbert_cache.items():
                json_serializable_sbert_cache[k] = v.tolist()
            print('Total of {} papers in sbert_cache'.format(len(json_serializable_sbert_cache)))
            with open('cache/sbert_cache.json', 'w') as file:
                print('Saving sbert_cache...')
                json.dump(json_serializable_sbert_cache, file)


            avg_precision_subdict = {}
            avg_ndcg_subdict = {}

            print('\n======================== Final Result for Year {} ========================'.format(test_year))
            for k in self.all_k:
                avg_precision = round(sum(top_k_precision.get(k))/len(top_k_precision.get(k)), 4)
                avg_precision_subdict[k] = [avg_precision]
                avg_ndcg = round(sum(top_k_ndcg.get(k))/len(top_k_ndcg.get(k)), 4)
                avg_ndcg_subdict[k] = [avg_ndcg]
                print('Precision@{}: {} || NDCG@{}: {}'.format(k, round(avg_precision, 4), k, round(avg_ndcg, 4)))
            print('===========================================================================')
            print()

            self.avg_precision_dict[test_year] = avg_precision_subdict
            self.avg_ndcg_dict[test_year] = avg_ndcg_subdict

    def get_embedding(self, paper_id):
        paper_text = self.paper_abstract.get(paper_id)
        if paper_text == None:
            paper_text = ['']
        paper_keywords = self.paper_keywords.get(paper_id)
        if paper_keywords == None:
            paper_keywords = ['']
        paper_abstract = sent_tokenize(paper_text[0].replace('(', '').replace(')', '').replace(':', ' '))
        paper_keywords_abstract = paper_abstract + paper_keywords
        paper_embed = self.string_vector_bert_cache(paper_id, paper_keywords_abstract)
        return paper_embed

    def string_vector_bert_cache(self, paper_id, list_of_sentences, mode = 'average'):
        if paper_id not in self.sbert_cache:
            if mode == 'average':
                embedding = np.mean(self.sbert.encode(list_of_sentences), axis = 0)
            else:
                pass
            self.sbert_cache[paper_id] = embedding
        else:
            embedding = self.sbert_cache.get(paper_id)
        return embedding

    def find_sim_cache(self, text_id1, text_emb1, text_id2, text_emb2, allow_perfect_sim = True):
        '''
        text_id1 = gold paper id / cluster paper id / recommended paper id
        text_id2 = data paper id / input paper id / gold paper id
        allow_perfect_sim = allow similarity to equal to 1. Set this to False when getting recommended papers (i.e., relevant papers based on the input papers)
        '''
        key = '({}, {})'.format(text_id1, text_id2)
        if key not in self.sim_cache:
            try:
                sim = round(1 - spatial.distance.cosine(text_emb1, text_emb2), 5)
            except Exception as e: 
                sim = 0
            if not allow_perfect_sim:
                if sim == 1:
                    sim = 0
                else:
                    pass
        else:
            sim = self.sim_cache.get(key)
        return sim

    def text_clean(self, text):
        stop_words = self.stop_words
        tokenizer = self.tokenizer
        stemmer = self.stemmer
        text = text.lower()
        text_tokenized = tokenizer.tokenize(text) 
        text_cleaned = [word for word in text_tokenized if not word in stop_words]
        text_stemmed = [stemmer.stem(word) for word in text_cleaned]
        return text_stemmed