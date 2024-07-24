import torch
import json, csv, os, random
import hashlib
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline 
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
import pandas as pd
from p_tqdm import p_map
from tqdm import tqdm
from itertools import islice
from node2vec import Node2Vec 
from copy import deepcopy
from gensim.models import Word2Vec 
import re
import math
import random
import os
from glob import glob
from os.path import exists
from scipy.spatial import distance
from scipy import spatial
from nltk import tokenize, skipgrams
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from fastbm25 import fastbm25
from itertools import chain, combinations
import copy
from copy import deepcopy
import gc # garbage collector
from cdlib import evaluation
import networkx.algorithms.community as nx_comm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import yake
import logging
import pickle
import gc
import ijson, bigjson

def giant_json_list(json_files):
    '''
    Reads a list of json file paths
    '''
    return [read_json_file(path) for path in tqdm(json_files)]

def read_json_file(path_to_file, array = False):
    output = {}
    with open(path_to_file,'r',buffering=100000) as f:
        data = bigjson.load(f)
        data_iter = data.iteritems()
        for item in tqdm(data_iter):
            key, value = item[0], item[1].to_python()

    return output

def get_hfembedding(list_of_texts, hf_model = 'gpt2', dimentionality_reductor = 'tsne', target_dimension = 300):
    feature_extractor = pipeline('feature-extraction', framework = 'pt', model = hf_model)
    cached_embeddings = {} 
    returned_embeddings = {}

    filename = 'cache/{}_embedding_cache.json'.format(hf_model)
    if exists(filename):
        cached_embeddings = {}
        for k, v in tqdm(ijson.kvitems(open(filename), '')):
            cached_embeddings[k] = v
        all_embeddings = []
        all_paper_id = []
        for k, v in cached_embeddings.items():
            all_embeddings.append(v)
            all_paper_id.append(k)
    else:
        all_embeddings = []
        all_paper_id = []
        if hf_model == 'gpt2':
            embed_size = 768
        else:
            pass
        for text_tup in tqdm(list_of_texts):
            paper_id = text_tup[0]
            try:
                x = feature_extractor(text_tup[-1])[0][0]
            except:
                x = [0] * embed_size
            all_embeddings.append(x)
            all_paper_id.append(paper_id)
            cached_embeddings[paper_id] = x
        print('Saving {} embeddings as chache\n'.format(hf_model))
        json.dump(cached_embeddings, open(filename, 'w'))

    if dimentionality_reductor == 'tsne':
        print('\nTraining TSNE on {} embedding...\n'.format(hf_model))
        dr = TSNE(n_components = target_dimension, perplexity = 5, method ='exact', learning_rate = 'auto', init= 'random')
    else: 
        pass

    reduced_embedding = dr.fit_transform(np.array(all_embeddings))
    reduced_embedding_tolist = reduced_embedding.tolist()
    for j in range(len(reduced_embedding)):
        returned_embeddings[all_paper_id[j]] = reduced_embedding_tolist[j]

    return returned_embeddings

def huggingface_feature_extrator(all_paper_text, dimentionality_reductor = 'tsne', target_dimension = 300, hf_model = 'gpt2'):
    '''
    > all_paper_text = text of each paper in the data in a dict format (paper_abstract from main.py)
    > dimentionality_reductor = dimentionality reduction method to use, by default it is using tsne
    > target_dimension = dimension for the final text embed
    '''
    all_texts = [] 
    all_embeddings = []
    for k, v in tqdm(all_paper_text.items()):
        curr_text = []
        for item in v:
            if type(item) != list:
                if item != '':
                    curr_text.append(item)
                else:
                    pass
            else:
                for itm in item:
                    curr_text.append(itm)
        all_texts.append((k, '. '.join(curr_text)))

    final_embeddings = get_hfembedding(all_texts, hf_model = hf_model, dimentionality_reductor = dimentionality_reductor, target_dimension = target_dimension)

    return final_embeddings

def fifty_fifty_data_splitter(x_data, y_data, authors_data, x_pos_data = None, randomize = False):

    '''
    outputs 2 sets of training and testing data where the second set is training data as test data and testing data as train data
    '''
    try:
        assert len(x_data) == len(y_data)
    except Exception as e:
        print(e)
        exit()

    assert len(x_data) == len(authors_data)
    if x_pos_data != None:
        assert len(x_data) == len(x_pos_data)
    else:
        pass

    total_data = len(x_data)
    if randomize:
        data_indexes = random.sample(range(0, total_data), total_data)
    else:
        data_indexes = [i for i in range(total_data)]

    firsthalf = round(len(x_data) / 2)
    data_indexes_train = data_indexes[:firsthalf]
    data_indexes_test = data_indexes[firsthalf:]

    assert len(data_indexes_train) + len(data_indexes_test) == len(data_indexes)

    # first set of data
    x_train_1 = [x_data[indx] for indx in data_indexes_train]
    x_train_pos_1 = [x_pos_data[indx] for indx in data_indexes_train]
    authors_train_1 = [authors_data[indx] for indx in data_indexes_train]
    y_train_1 = [y_data[indx] for indx in data_indexes_train]

    x_test_1 = [x_data[indx] for indx in data_indexes_test]
    x_test_pos_1 = [x_pos_data[indx] for indx in data_indexes_test]
    authors_test_1 = [authors_data[indx] for indx in data_indexes_test]
    y_test_1 = [y_data[indx] for indx in data_indexes_test]

    # second set of data
    x_test_2 = deepcopy(x_train_1)
    x_test_pos_2 = deepcopy(x_train_pos_1)
    authors_test_2 = deepcopy(authors_train_1)
    y_test_2 = deepcopy(y_train_1)

    x_train_2 = deepcopy(x_test_1)
    x_train_pos_2 = deepcopy(x_test_pos_1)
    authors_train_2 = deepcopy(authors_test_1)
    y_train_2 = deepcopy(y_test_1)

    assert x_test_2 == x_train_1
    assert x_test_pos_2 == x_train_pos_1
    assert authors_test_2 == authors_train_1
    assert y_test_2 == y_train_1

    assert x_train_2 == x_test_1
    assert x_train_pos_2 == x_test_pos_1
    assert authors_train_2 == authors_test_1
    assert y_train_2 == y_test_1

    return x_train_1, x_train_pos_1, authors_train_1, y_train_1, x_test_1, x_test_pos_1, authors_test_1, y_test_1, x_train_2, x_train_pos_2, authors_train_2, y_train_2, x_test_2, x_test_pos_2, authors_test_2, y_test_2


def load_randomization(filename):
    unserialized_data = json.load(open('{}.json'.format(filename)))
    unserialized_ndarray = {}
    for k, v in unserialized_data.items():
        unserialized_ndarray[k] = np.array(v)
    return unserialized_ndarray

def batch_sampling_cf(batch_size, authors, neighbors_reps, inputs, inputs_position, labels, shuffling = True):
    batch_authors = []
    batch_neighbors_rep = []
    batch_inputs = []
    batch_inputs_pos = []
    batch_labels = []
    all_batches_authors = []
    all_batches_neighbors_rep = []
    all_batches_inputs = []
    all_batches_inputs_pos = []
    all_batches_labels = []

    assert len(inputs) == len(inputs_position)

    if isinstance(authors, tuple):
        authors = authors[0]
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    idx_list = [i for i in range(len(authors))]
    if shuffling:
        random.seed(123)
        random.shuffle(idx_list)

    counter = 0
    for i in range(len(authors)):
        batch_authors.append(authors[idx_list[i]])
        batch_neighbors_rep.append(neighbors_reps[idx_list[i]])
        batch_inputs.append(inputs[idx_list[i]])
        batch_inputs_pos.append(inputs_position[idx_list[i]])
        try:
            assert len(inputs[idx_list[i]]) == len(inputs_position[idx_list[i]])
        except AssertionError as e:
            for n in range(len(inputs)):
                print(inputs[n], len(inputs[n]))
                print(inputs_position[n], len(inputs_position[n]))
                print()
        batch_labels.append(labels[idx_list[i]])
        counter += 1
        if (counter % batch_size == 0) or (i == len(authors) - 1):
            all_batches_authors.append(batch_authors)
            all_batches_neighbors_rep.append(batch_neighbors_rep)
            all_batches_inputs.append(batch_inputs)
            all_batches_inputs_pos.append(batch_inputs_pos)
            all_batches_labels.append(batch_labels)
            batch_authors = []
            batch_neighbors_rep = []
            batch_inputs = []
            batch_inputs_pos = []
            batch_labels = []

    assert len(all_batches_authors) == len(all_batches_neighbors_rep)
    assert len(all_batches_neighbors_rep) == len(all_batches_inputs)
    assert len(all_batches_inputs) == len(all_batches_inputs_pos)
    assert len(all_batches_inputs_pos) == len(all_batches_labels)

    return all_batches_authors, all_batches_neighbors_rep, all_batches_inputs, all_batches_inputs_pos, all_batches_labels

# this is for batching
def batch_sampling(batch_size, authors, inputs, inputs_position, labels, shuffling = True):
    batch_authors = []
    batch_inputs = []
    batch_inputs_pos = []
    batch_labels = []
    all_batches_authors = []
    all_batches_inputs = []
    all_batches_inputs_pos = []
    all_batches_labels = []

    try:
        assert len(inputs) == len(inputs_position)
    except Exception as e:
        print(e)
        exit()

    if isinstance(authors, tuple):
        authors = authors[0]
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    idx_list = [i for i in range(len(authors))]
    if shuffling:
        random.seed(123)
        random.shuffle(idx_list)

    counter = 0
    for i in range(len(authors)):
        batch_authors.append(authors[idx_list[i]])
        batch_inputs.append(inputs[idx_list[i]])
        batch_inputs_pos.append(inputs_position[idx_list[i]])
        batch_labels.append(labels[idx_list[i]])
        counter += 1
        if (counter % batch_size == 0) or (i == len(authors) - 1):
            all_batches_authors.append(batch_authors)
            all_batches_inputs.append(batch_inputs)
            all_batches_inputs_pos.append(batch_inputs_pos)
            all_batches_labels.append(batch_labels)
            batch_authors = []
            batch_inputs = []
            batch_inputs_pos = []
            batch_labels = []

    return all_batches_authors, all_batches_inputs, all_batches_inputs_pos, all_batches_labels

def get_node_embeddings(graph,
                          node_type,
                          author_features, author_decoder,
                          paper_features, paper_decoder,
                          topic_features = '', topic_decoder = ''
                          ):
    '''
    Returns a tensor with dimension [number_of_nodes, embedding_dimension]

    takes the following:
    > graph = DGL heterograph
    > node_type = current node type that is going to be processed
    > author_features = all random author node inits
    > author_decoder = from BuildKG for hetero graph, decodes the current node index into its worded index
    > paper_features = all random paper node inits
    > paper_decoder = ** same like author_decoder but for paper nodes **
    > topic_features = all random topic node inits ## TO DO ##
    > topic_decoder = ** same like author_decoder but for topics ** ## TO DO ##
    '''

    list_of_tensors = []

    list_of_idx = graph.nodes(node_type)

    for idx in list_of_idx:

        key = idx.item()

        if node_type == 'author':
            decoded_idx = author_decoder.get(key)
            embedding = author_features.get(decoded_idx)
        if node_type == 'paper':
            decoded_idx = paper_decoder.get(key)
            embedding = paper_features.get(decoded_idx)
        if node_type == 'topic':
            decoded_idx = topic_decoder.get(key)
            embedding = topic_features.get(decoded_idx)

        embedding = torch.from_numpy(embedding).type(torch.float)
        list_of_tensors.append(embedding[0])

    returned_stack_of_tensors = torch.stack(list_of_tensors)

    return returned_stack_of_tensors

def communities_scoring(g, communities, communities_nx, graph_name = None):
    '''
    community scoring to evaluate the graph communities in clusterer.py

    g = networkx graph
    communities = resulting communities from the community detection algorithm from cdlib
    communities_nx = resulting communities from networkx communities detection algorithm
    graph_name = the graph name, should either be original/giant components/random rewiring
    '''
    print('Communities scoring for {} graph'.format(graph_name))
     # internal connectivity
    internal_density_scr = evaluation.internal_edge_density(g, communities)
    print('1) internal_density_scr =', internal_density_scr)
    edges_inside_scr = evaluation.edges_inside(g, communities)
    print('2) edges_inside_scr =', edges_inside_scr)
    average_degree_scr = evaluation.average_internal_degree(g, communities)
    print('3) average_degree_scr =', average_degree_scr)
    fomd_scr = evaluation.fraction_over_median_degree(g, communities)
    print('4) fomd_scr =', fomd_scr)

    expansion_scr = evaluation.expansion(g, communities)
    print('5) expansion_scr =', expansion_scr)
    cut_ratio_scr = evaluation.cut_ratio(g, communities)
    print('6) cut_ratio_scr =', cut_ratio_scr)

    conductance_scr = evaluation.conductance(g, communities)
    print('7) conductance_scr =', conductance_scr)
    normalized_cut_scr = evaluation.normalized_cut(g, communities)
    print('8) normalized_cut_scr =', normalized_cut_scr)
    max_odf_scr = evaluation.max_odf(g, communities)
    print('9) max_odf_scr =', max_odf_scr)
    avg_odf = evaluation.avg_odf(g, communities)
    print('10) avg_odf =', avg_odf)
    flake_odf = evaluation.flake_odf(g, communities)
    print('11) flake_odf =', flake_odf)

    erdos_renyi_modularity_scr = evaluation.erdos_renyi_modularity(g, communities)
    print('12) erdos_renyi_modularity_scr =', erdos_renyi_modularity_scr)
    link_modularity_scr = evaluation.link_modularity(g, communities)
    print('13) link_modularity_scr =', link_modularity_scr)
    modularity_density_scr = evaluation.modularity_density(g, communities)
    print('14) modularity_density_scr =', modularity_density_scr)
    newman_girvan_modularity_scr = evaluation.newman_girvan_modularity(g, communities)
    print('15) newman_girvan_modularity_scr =', newman_girvan_modularity_scr)
    z_modularity_scr = evaluation.z_modularity(g, communities)
    print('16) z_modularity_scr =', z_modularity_scr)

    nx_modularity_scr = nx_comm.modularity(g, communities_nx)
    print('17) Networkx modularity =', nx_modularity_scr)

    print('\n ==== End of {} graph community scoring ====\n'.format(graph_name))

def NLLLoss(logs, targets):
    '''
    takes softmax results and classification labels (y) as input and outputs loss scores
    '''
    out = torch.zeros_like(targets, dtype = torch.float)
    for i in range(len(targets)):
    safe_tensor = torch.where(torch.isnan(out), torch.zeros_like(out), out)
    out = safe_tensor
    sum_out = out.sum()
    len_out = len(out)
    result = -sum_out/len_out
    return result

class NeighborsInclusion():
    '''
    Include the embedding of paper keywords' ontology neighbors
    '''
    def __init__(self,
                 paper_keywords,
                 keyword_neighbors,
                 distance = 3, # distance is used only to create the cache file later
                 ):
        '''
        > abstract_embeddings: dictionary of existing abstract embeddings. format is {paper_id: abstract embedding (type list)}
        > paper_keywords: dictionary of paper keywords. format is {paper_id: [list of paper keywords (including the ones extracted by YAKE)]}
        > keyword_neighbors: dictionary of neighbors of each paper keyword. format is {paper_keyword: [list of neighbors in the ontology]}
        > distance: the distance used to obtain keyword_neighbors. Default is set to 3.
        '''
        filename = 'cache/keyword_neighbors_embedding_per_paper_using_distance_{}.json'.format(distance)
        self.paper_keywords = paper_keywords
        self.distance = distance
        self.keyword_embeddings = {} # cache while creating keyword embeddings
        try:
            self.kw_neighbor_embeddings = json.load(open(filename))
        except Exception as e:
            print(e)
            self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.kw_neighbor_embeddings = self.get_keyword_neighbor_embeddings( keyword_neighbors, filename)
        assert len(self.kw_neighbor_embeddings) == len(paper_keywords)

    def get_keyword_neighbor_embeddings(self, keyword_neighbors, filename):
        updated_abstract_embedding = {}
        counter = 0
        for k, v in tqdm(self.paper_keywords.items()):
            keywords = v
            keyword_embeddings = 0
            for kw in keywords:
                neighbors = keyword_neighbors.get(kw)
                if len(neighbors) == 1:
                    keyword_embeddings = self.sbert.encode([neighbors[0]])
                else:
                    for neighbor in neighbors:
                        if isinstance(keyword_embeddings, int):
                            if neighbor not in self.keyword_embeddings:
                                keyword_embeddings = self.sbert.encode([neighbor])
                                self.keyword_embeddings[neighbor] = keyword_embeddings
                            else:
                                keyword_embeddings = self.keyword_embeddings.get(neighbor)
                        else:
                            if neighbor not in self.keyword_embeddings:
                                neighbor_embedding = self.sbert.encode([neighbor])
                                self.keyword_embeddings[neighbor] = neighbor_embedding
                            else:
                                neighbor_embedding = self.keyword_embeddings.get(neighbor)
                            keyword_embeddings = self.combine_embeddings(keyword_embeddings, neighbor_embedding)
            if isinstance(keyword_embeddings, int):
                updated_abstract_embedding[k] = None
            else:
                keyword_embeddings = keyword_embeddings.tolist()[0]
                updated_abstract_embedding[k] = keyword_embeddings
            counter += 1
            if counter % 1000 == 0:
                json.dump(updated_abstract_embedding, open(filename, 'w'))
        json.dump(updated_abstract_embedding, open(filename, 'w'))

    def combine_embeddings(self, old_embedding, current_embedding):
        combined_embedding = np.mean([old_embedding, current_embedding], axis = 0)
        return combined_embedding

    def include_kw_neighbors_embeddings(self, abstract_embeddings):
        filename = 'cache/abstract_embedding_updated_with_keyword_neighbors_with_distance{}.json'.format(self.distance)
        try:
            updated_abstract_embedding = json.load(open(filename))
        except:
            updated_abstract_embedding = {}
            print('Combining keyword neighbor embedding with the abstract embedding')
            for k, v in tqdm(abstract_embeddings.items()):
                abstract_embedding = v
                keywords = self.paper_keywords.get(k)
                if keywords is not None:
                    curr_keyword_embedding = 0
                    for keyword in keywords:
                        keyword_embedding = self.kw_neighbor_embeddings.get(k)
                        if keyword_embedding is not None:
                            keyword_embedding = np.array(keyword_embedding)
                            if isinstance(curr_keyword_embedding, int):
                                abstract_embedding = np.array(abstract_embedding)
                                curr_keyword_embedding = np.mean([abstract_embedding, keyword_embedding], axis = 0)
                            else:
                                curr_keyword_embedding = np.mean([curr_keyword_embedding, keyword_embedding], axis = 0)
                if isinstance(curr_keyword_embedding, int): # did not change (does not have keywords)
                    updated_abstract_embedding[k] = abstract_embedding
                elif isinstance(curr_keyword_embedding, list):
                    updated_abstract_embedding[k] = curr_keyword_embedding
                else:
                    curr_keyword_embedding = curr_keyword_embedding.tolist()
                    updated_abstract_embedding[k] = curr_keyword_embedding
            print('Saving the updated abstract embedding...\n')
            json.dump(updated_abstract_embedding, open(filename, 'w'))
        return updated_abstract_embedding

class DiversityCacheBuilder():
    '''
    Build diversity cache between each keyword that exist in the data
    This chache builder also updates the keywords data from paper if required
    '''
    def __init__(self,
                 topic_and_level_info,
                 hops_matrix,
                 update_paper_keywords = False,
                ):

        self.paper_abstract = json.load(open('../data/paper_paper_abstract_title_bert.json'))

        if update_paper_keywords:
            self.paper_keywords = {}
            kw_extractor = yake.KeywordExtractor()
            language = 'en'
            max_ngram_size = 4
            deduplication_threshold = 0.9
            num_of_keywords_abstract = 4
            num_of_keywords_title = 4
            stop_words = set(stopwords.words('english'))

            custom_kw_extractor_abstract = yake.KeywordExtractor(lan = language, n = max_ngram_size, dedupLim = deduplication_threshold, top = num_of_keywords_abstract, features = None)
            custom_kw_extractor_title = yake.KeywordExtractor(lan = language, n = max_ngram_size, dedupLim = deduplication_threshold, top = num_of_keywords_title, features = None)

            for paper_id, all_texts in tqdm(self.paper_abstract.items()):
                abstract, title, keywords = all_texts
                abstract = list(abstract.split())
                title = list(title.split())
                cleaned_abstract = [word for word in abstract if not word.lower() in stop_words]
                cleaned_title = [word for word in title if not word.lower() in stop_words]
                abstract = ' '.join(cleaned_abstract)
                title = ' '.join(cleaned_title)
                unique_keywords = set(keywords)
                keywords_from_abstract = custom_kw_extractor_abstract.extract_keywords(abstract)
                keywords_from_title = custom_kw_extractor_title.extract_keywords(title)
                for item in keywords_from_abstract:
                    unique_keywords.add(item[0])
                for item in keywords_from_title:
                    unique_keywords.add(item[0])
                self.paper_keywords[paper_id] = list(unique_keywords)
        else:
            self.paper_keywords = json.load(open('../data/paper_corresponding_topics_yake.json'))

        try:
            self.diversity_matrix = pd.read_pickle('cache/diversity_cache.pkl')
        except: # no diversity matrix cache
            print('Creating diversity cache...')
            all_unique_keywords = set()
            for k, v in tqdm(self.paper_keywords.items()):
                for item in v:
                    all_unique_keywords.add(item)
            print('There are {} unique keywords...'.format(len(all_unique_keywords)))
            all_unique_keywords = list(all_unique_keywords)
            all_unique_keywords_pairs = self.get_all_possible_pairs(all_unique_keywords)
            all_unique_keywords_pairs = list(all_unique_keywords_pairs)
            for item in tqdm(all_unique_keywords):
                same_item_pair = (item, item)
                all_unique_keywords_pairs.add(same_item_pair)
            print('There are {} pairs to process...\n'.format(len(all_unique_keywords_pairs)))

            diversity_zeros = np.zeros((len(all_unique_keywords), len(all_unique_keywords)))
            self.diversity_matrix = pd.DataFrame(diversity_zeros, index = all_unique_keywords, columns = all_unique_keywords)

            for pair in tqdm(all_unique_keywords_pairs):
                source, target = pair
                try:
                    hops = hops_matrix.loc[kwd_inpt, kwd_rec]
                except:
                    hops = -1
                source_level_parent = topic_and_level_info.get(source)
                target_level_parent = topic_and_level_info.get(target)
                if source_level_parent is None or target_level_parent is None:
                    diversity_score = math.inf # infinity
                else:
                    try:
                        levels_source, parents_source = [], []
                        levels_target, levels_source = [], []
                        for item in source_level_parent:
                            level, parent = item
                            levels_source.append(level)
                            parents_source.append(parent)
                        for item in target_level_parent:
                            level, parent = item
                            levels_target.append(level)
                            parents_target.append(parent)
                        levels_source, parents_source = self.level_parent_sort(levels_source, parents_source)
                        levels_target, parents_target = self.level_parent_sort(levels_target, parents_target)
                        if len(levels_target) > 1:
                            for i in range(len(levels_target)):
                                if levels_target[i] in levels_source:
                                    try:
                                        target_level = levels_target[i]
                                        target_parent = parents_target[i]
                                        source_parent = parents_source[i]
                                    except:
                                        target_level = levels_target[0]
                                        target_parent = parents_target[0]
                                        source_parent = parents_source[0]
                                else:
                                    target_level = levels_target[0]
                                    target_parent = parents_target[0]
                                    source_parent = parents_source[0]
                                if source_parent != target_parent:
                                    parent_statement = 1
                                else:
                                    parent_statement = 0
                                parent_statement_weight = self.sigmoid(parent_statement)
                                diversity_score = (hops / (2^(abs(levels_source - levels_target)))) * parent_statement_weight
                                # diversity_score = (hops / (2^(abs(levels_source - levels_target)))) + parent_statement_weight
                    except:
                        pass
                    self.diversity_matrix[source][target] = diversity_score
                    self.diversity_matrix[target][source] = diversity_score

            self.diversity_matrix.to_pickle('cache/diversity_cache.pkl') # pickling the cache

    def level_parent_sort(self, levels_list, parents_list):
        '''
        sorting the levels list and use the same index for returning the parents list
        '''
        levels_list = [(item, levels_list.index(item)) for item in levels_list]
        levels_list.sort(key = lambda tup: tup[0])
        returned_levels = [item[0] for item in levels_list]
        returned_parents = [parents_list[item[-1]] for item in levels_list]
        return returned_levels, returned_parents

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def get_all_possible_pairs(self, lst):
        pairs = []
        for item1 in tqdm(lst):
            for item2 in lst:
                if item1 == item2:
                    pass
                else:
                    pair = (item1, item2)
                    pair_inversed = (item2, item1)
                    if pair_inversed in pairs:
                        pass
                    else:
                        pairs.append(pair)
        return pairs

class OntologyBuilder():
    '''
    Builds ontology in networkx format through a loaded .csv file from past research (the ontology extension research result)
    > This class can also remove unimportant terms when required
    '''
    def __init__(self,
                 paper_abstract,
                 paper_and_year,
                 ontox_classifier,
                 ontox_status,
                 test_year,
                 device,
                 train_year_range = 5, # setting train data year range as 5 by default
                 dynamic_onto = False, # dynamic onto run
                 vanilla_cso = False,
                 onto_important_terms_only = False,
                 term_importance = 'bm25', # can choose either tfidf or bm25
                 document_lemmatization = False,
                 bm25_representative_measure = 'average',
                 make_onto_topic_hops_matrix = False,
                 paper_keywords = None,
                 simple_onto_extend = False,
                 similarity_for_onto_extend = False,
                 dynamic_onto_extend = False, # dynamic ontology extension
                 triple_types_experiment = False,
                 ):

        self.onto_important_terms_only = onto_important_terms_only
        self.term_importance = term_importance

        self.ontology_edges = {}
        self.ontology_parent_children, self.ontology_child_parents = {}, {}
        self.ontology_node_levels = {}
        self.unimportant_terms = set()
        self.similarity_for_onto_extend = similarity_for_onto_extend

        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        self.sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.device = device

        if triple_types_experiment:
            triple_type = int(input('Triple type to use: '))
        else:
            triple_type = -1

        self.gen_kwargs = {
                "max_length": 256,
                "length_penalty": 0,
                "num_beams": 3,
                "num_return_sequences": 1,
                }

        if self.onto_important_terms_only:
            stop_words = set(stopwords.words('english'))
            if document_lemmatization:
                lemmatizer = WordNetLemmatizer()

        if dynamic_onto:
            filename = '../../althubaiti_vs_me/{}_status_{}_extended_ontology.csv'.format(ontox_classifier, ontox_status)
        else:
            filename = '../data/onto_dictionary_nolimit.csv'
        print('Using this version of ontology: {}\n'.format(filename))
        csv_file = open(filename)
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for row in csv_reader:
            child, parent = row[0].replace('_', ' '), row[-1].replace('_', ' ')
            if parent not in self.ontology_edges:
                self.ontology_edges[parent] = [child]
            else:
                self.ontology_edges[parent] += [child]
            if parent not in self.ontology_parent_children:
                self.ontology_parent_children[parent] = [child]
            else:
                self.ontology_parent_children[parent] += [child]
            if child not in self.ontology_child_parents:
                self.ontology_child_parents[child] = [parent]
            else:
                self.ontology_child_parents[child] += [parent]

        self.topic_levels = self.generate_topic_levels_and_parents(self.ontology_parent_children)

        keywords_to_remove = set()
        ontology_topics_per_year = {} # format is {year:[topic]}
        if dynamic_onto:
            for k, v in self.ontology_edges.items():
                if vanilla_cso:
                    try:
                        if int(k) < -math.inf:
                            print(k, 'kept')
                            ontology_topics_per_year[k] = v
                            pass
                        else:
                            print(k, 'removed')
                            ontology_topics_per_year[k] = v
                            for item in v:
                                keywords_to_remove.add(item)
                    except:
                        pass
                else:
                    try:
                        if int(k) < test_year:
                            print(k, 'kept')
                            ontology_topics_per_year[k] = v
                            pass
                        elif int(k) == test_year:
                            print(k, 'kept')
                            ontology_topics_per_year[k] = v
                            pass
                        else:
                            print(k, 'removed')
                            ontology_topics_per_year[k] = v
                            for item in v:
                                keywords_to_remove.add(item)
                    except:
                        pass
                json.dump(ontology_topics_per_year, open('cache/onto_topics_per_year.json', 'w'))
        else:
            try:
                ontology_topics_per_year = json.load(open('cache/onto_topics_per_year.json'))
            except:
                print('Cache file onto_topics_per_year.json does not exist. This file can be obtained during dynamic ontology run.')

        if self.onto_important_terms_only:
            print('Getting important terms only from the ontology...\n')
            unique_ontology_terms = set()
            paper_id_and_corpus = {}
            for k, v in tqdm(paper_abstract.items()):
                abstract, keywords = v[0], v[-1]
                combined_text = abstract + ' ' + ' '.join(keywords)
                paper_id_and_corpus[k] = combined_text
            ontology_terms = set()
            unimportant_terms = set()
            for k, v in ontology_topics_per_year.items():
                try:
                    k = int(k)
                    for item in v:
                        ontology_terms.add((item, k))
                except Exception as e:
                    pass
            if self.term_importance == 'tfidf':
                print('\nGetting important terms via TF-IDF...\n')
                term_counts = {} 
                term_tfidf = {}
                print('Counting terms in corpuses...\n')
                for item in tqdm(ontology_terms):
                    term, year = item
                    total_term_count = 0
                    total_doc_containing_term = 0
                    all_tfs = []
                    for paper_id, document_temp in paper_id_and_corpus.items():
                        term_count = document_temp.count(term)
                        document_tokens = word_tokenize(document_temp)
                        document_cleaned = [w for w in document_tokens if not w in stop_words]
                        if document_lemmatization:
                            document_cleaned = [lemmatizer.lemmatize(w) or w in document_cleaned] 
                        document = ' '.join(document_cleaned)
                        document_word_count = len(list(document.split()))
                        try:
                            curr_tf_score = term_count/document_word_count 
                        except: 
                            curr_tf_score = 0
                        all_tfs.append(curr_tf_score)
                        total_term_count += term_count
                        if term_count > 0:
                            total_doc_containing_term += 1
                        else:
                            self.unimportant_terms.add(term)

                        try:
                            idf_score = math.log(len(paper_id_and_corpus)/total_doc_containing_term)
                        except:
                            idf_score = 0
                        for tf_score in all_tfs:
                            tfidf_score = tf_score * idf_score
                            if term not in term_tfidf:
                                term_tfidf[term] = [tfidf_score]
                            else:
                                term_tfidf[term] += [tfidf_score]
                term_tfidf_combined = {}
                for k, v in tqdm(term_tfidf.items()):
                    term_tfidf_combined[k] = sum(v)
                temp_term_tfidf = sorted(term_tfidf_combined.items(), key = lambda x:x[1], reverse = True)
                term_tfidf_sorted = dict(temp_term_tfidf)
                for k, v in term_tfidf_sorted.items():
                    if v < 0.1:
                        self.unimportant_terms.add(k)
                    else:
                        pass
            elif self.term_importance == 'bm25':
                print('\nGetting important terms via BM25...\n')
                all_term_bm25 = {} 
                corpus = [doc for doc in paper_id_and_corpus.values()]
                tokenized_corpus_temp = [doc.lower().split(" ") for doc in corpus] 
                tokenized_corpus = []
                for doc in tokenized_corpus_temp:
                    tokenized_corpus.append([w for w in doc if not w in stop_words])
                if document_lemmatization:
                    tokenized_corpus_temp = deepcopy(tokenized_corpus)
                    tokenized_corpus = []
                    for doc in tokenized_corpus_temp:
                        tokenized_corpus.append([lemmatizer.lemmatize(w) for w in doc])
                bm25_model = fastbm25(tokenized_corpus)
                for item in tqdm(ontology_terms):
                    query, year = item
                    result = bm25_model.top_k_sentence(query, k = len(corpus))
                    result_score_only = [item[-1] for item in result]
                    term = query
                    if term not in all_term_bm25:
                        all_term_bm25[term] = result_score_only
                    else:
                        all_term_bm25[term] += result_score_only
                all_term_bm25_combined = {}
                for k, v in tqdm(all_term_bm25.items()):
                    try:
                        if bm25_representative_measure == 'average':
                            all_term_bm25_combined[k] = round(sum(v)/len(v), 4)
                        elif bm25_representative_measure == 'sum':
                            all_term_bm25_combined[k] = sum(v)
                        else:
                            print('BM25 Representative measure {} does not apply. Use either average or sum only'.format(bm25_representative_measure))
                            exit()
                    except:
                        all_term_bm25_combined[k] = 0
                    temp_term_bm25 = sorted(all_term_bm25_combined.items(), key=lambda x:x[1], reverse=True)
                    term_bm25_sorted = dict(temp_term_bm25)
                    if bm25_representative_measure == 'average':
                        threshold = 9
                    elif bm25_representative_measure == 'sum':
                        threshold = 45000
                    else:
                        print('BM25 Representative measure {} does not apply. Use either average or sum only'.format(bm25_representative_measure))
                        exit()
                    for k, v in term_bm25_sorted.items():
                        if v < threshold:
                            self.unimportant_terms.add(k)
                        else:
                            pass

        if dynamic_onto:
            if self.onto_important_terms_only:
                for item in tqdm(unimportant_terms):
                    keywords_to_remove.add(item)
            else:
                pass
            keywords_to_remove = list(keywords_to_remove)
            temp_dict = {}
            for k, v in self.ontology_edges.items():
                try:
                    int(k)
                    pass
                except:
                    for item in v:
                        if item not in keywords_to_remove:
                            if k not in temp_dict:
                                temp_dict[k] = [item]
                            else:
                                temp_dict[k] += [item]
                        else:
                            pass
            assert temp_dict != self.ontology_edges
            self.ontology_edges = temp_dict
        else:
            pass

        self.ontology_graph = nx.Graph(self.ontology_edges) 
        self.ontology_graph_directed = nx.DiGraph(self.ontology_edges)

        if simple_onto_extend:
            self.ontology_graph_extended = deepcopy(self.ontology_graph)
            try:
                self.triples = [] 
                self.triples_per_paper = {}
                with open('cache/triples_per_paper.json', 'r', buffering = 100000) as f:
                    self.triples_per_paper = json.load(f)

                print('Triples cache loaded...\n')
                for k, v in self.triples_per_paper.items():
                    for tripl in v:
                        source, relationship, target = tripl[0], tripl[1], tripl[-1]
                        clean_source, clean_target = self.text_cleaner(source), self.text_cleaner(target)
                        self.triples.append((clean_source, relationship, clean_target))
            except Exception as e:
                print(e)
                exit()
                print('\nGetting new triples from abstract using REBEL...')
                self.triples = []
                self.triples_per_paper = {} 
                all_abstracts_only = [item for item in list(paper_abstract.values()) if item != None] 
                all_paper_ids_only = [item for item in list(paper_abstract.keys()) if paper_abstract[item] != None]
                assert len(all_abstracts_only) == len(all_paper_ids_only)
                for i in tqdm(range(len(all_abstracts_only))):
                    text = all_abstracts_only[i][0]
                    paper_id = all_paper_ids_only[i]
                    self.generate_triples(text, paper_id)
                with open('cache/new_topic_triples.csv','w') as out:
                    csv_out = csv.writer(out)
                    csv_out.writerow(['source','relationship', 'target'])
                    for row in self.triples:
                        csv_out.writerow(row)
                json.dump(self.triples_per_paper, open('cache/triples_per_paper.json', 'w'))

            if dynamic_onto_extend:
                filtered_triples = []
                for k, v in self.triples_per_paper.items():
                    paper_year = paper_and_year.get(k)
                    if paper_year == None:
                        pass
                    else:
                        if paper_year < test_year:
                            filtered_triples.append(v[0])
                        else:
                            pass
                print('{} out of {} triples ({}%) are going to be used for this ontology extension...\n'.format(len(filtered_triples), len(self.triples_per_paper), round(len(filtered_triples)/len(self.triples_per_paper), 4) * 100))
                self.triples = filtered_triples

            success_extend = 0
            triples_remainder = []
            triples_len_before = len(self.triples)
            for item in tqdm(self.triples):
                source, target = item[-1], item[0]
                if source in self.ontology_graph.nodes() or target in self.ontology_graph.nodes():
                    self.ontology_graph_extended.add_edge(source, target)
                    success_extend += 1
                else:
                    triples_remainder.append(item)
            print('\n{} out of {} REBEL triples can be directly linked to the original CSO\n'.format(success_extend, len(self.triples)))
            if triple_type != 1:
                temp = []
                success_extend = 0
                for item in triples_remainder:
                    source, target = item[-1], item[0]
                    if source in self.ontology_graph_extended.nodes() or target in self.ontology_graph_extended.nodes():
                        self.ontology_graph_extended.add_edge(source, target)
                        success_extend += 1
                    else:
                        temp.append(item)

                print('{} out of {} REBEL triples can be directly linked to the CSO + REBEL extension graph\n'.format(success_extend, len(triples_remainder)))
                print('Extension complete\n')
            else:
                print('Extension using type 1 triples only complete\n')
                pass

        if make_onto_topic_hops_matrix:
            ontology_nodes = list(self.ontology_graph.nodes()) # this is all ontology topics
            try:
                self.onto_topic_hops_matrix = pd.read_pickle('cache/ontology_topic_hops_matrix_cache.pkl')
            except:
                hops_zeros = np.zeros((len(ontology_nodes), len(ontology_nodes)))
                self.onto_topic_hops_matrix = pd.DataFrame(hops_zeros, index = ontology_nodes, columns = ontology_nodes)
                all_possible_pairs = list(combinations(ontology_nodes, 2))
                print('CREATING ONTOLOGY TOPIC HOPS MATRIX...\n')
                for item in tqdm(all_possible_pairs):
                    source, target = item[0], item[-1]
                    source = self.is_topic_in_onto(item[0], ontology_nodes)
                    target = self.is_topic_in_onto(item[-1], ontology_nodes)
                    try:
                        hop_distance = len(nx.shortest_path(self.ontology_graph, source = source, target = target)) - 1 
                    except Exception as e: 
                        print(e, source, target)
                        input()
                        hop_distance = -1
                    self.onto_topic_hops_matrix[source][target] = hop_distance
                    self.onto_topic_hops_matrix[target][source] = hop_distance
                self.onto_topic_hops_matrix.to_pickle('cache/ontology_topic_hops_matrix_cache.pkl') # pickling the cache
                print('Ontology topic hops matrix cache created...\n')
        else:
            self.onto_topic_hops_matrix = None

        try:
            self.topic_matches = json.load(open('cache/topic_matches.json'))
        except:
            if paper_keywords == None:
                print('Making topic partial matches cache, need to declare paper keywords. Exiting...')
                exit()
            else:
                pass
            self.topic_matches = {}
            all_paper_topics = set()
            for k, v in tqdm(paper_keywords.items()):
                for item in v:
                    if item == '' or item == ' ':
                        pass
                    else:
                        all_paper_topics.add(item)
            zero_counter = 0
            for topic in tqdm(all_paper_topics):
                if topic in ontology_nodes:
                    self.topic_matches[topic] = [(topic, len(topic.split()))]
                else:
                    temp_set = set()
                    for onto_topic in ontology_nodes:
                        if topic in onto_topic: 
                            match_counter = 0 
                            for word in topic.split():
                                if word in onto_topic:
                                    match_counter += 1
                            temp_set.add((onto_topic, match_counter))
                        else:
                            pass
                    if len(temp_set) == 0:
                        self.topic_matches[topic] = [(topic, 0)] 
                        zero_counter += 1
                    elif len(temp_set) == 1:
                        self.topic_matches[topic] = [(topic, 1)] 
                    else:
                        self.topic_matches[topic] = list(temp_set)
            for k, v in self.topic_matches.items():
                print(k, v)
            print('There are {} out of {} topics that have zero matches'.format(zero_counter, len(self.topic_matches)))
            json.dump(self.topic_matches, open('cache/topic_matches.json', 'w'))

    def is_topic_in_onto(self, topic, ontology_topics):
        '''
        Check if a topic exists inside of the ontology, and if the topic does not exist from exact match, it should partial match the topic to any in the ontology.
        steps:
        > exact keywords match of the topic to the topics in the ontology
        > if no exact match is found, try partial match
            > if a single partial match is found then return this single partial match and change the topic into this partial match
            > if there are multiple partial matches then ...
        > if there is no partial match, then return the original keyword
        '''
        if topic in ontology_topics: 
            return topic
        else: 
            partial_matches = set()
            for onto_topic in ontology_topics:
                if topic in onto_topic:
                    partial_matches.add(topic)
                else:
                    pass
            if len(partial_matches) == 0 or len(partial_matches) == 1:
                return topic
            else: 
                print('Current topic', topic)
                print('Partial matches:', partial_matches)
                input()

    def get_embedding(self, topic):
        '''
        getting the SBERT embedding of topic
        '''
        if topic not in self.sbert_rebel_cso:
            embedding = np.mean(self.sbert.encode([topic]), axis = 0)
            self.sbert_rebel_cso[topic] = embedding
        else:
            embedding = self.sbert_rebel_cso.get(topic)
        return embedding

    def text_cleaner(self, text):
        x = text.replace(' - ', '-')
        x = x.replace('- ', '')
        x = x.replace(" '", '')
        return x

    def extract_triplets(self, text): # REBEL triplets extractor
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})

        return triplets

    def generate_triples(self, text, paper_id):
        model_inputs = self.tokenizer(text, max_length = 512, padding = True, truncation = True, return_tensors = 'pt')
        generated_tokens = self.model.generate(model_inputs["input_ids"].to(self.device), attention_mask = model_inputs["attention_mask"].to(self.device), **self.gen_kwargs)
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens = False)
        for idx, sentence in enumerate(decoded_preds):
            et = self.extract_triplets(sentence)
            for t in et:
                self.triples.append((t['head'], t['type'], t['tail']))
                # for paper_id in paper_ids:
                if paper_id not in self.triples_per_paper:
                    self.triples_per_paper[paper_id] = [(t['head'], t['type'], t['tail'])]
                else:
                    self.triples_per_paper[paper_id] += [(t['head'], t['type'], t['tail'])]

    def generate_topic_levels_and_parents(self, parent_children):
        '''
        returns the level information of each topic in the CSO in the form of dictionary {topic : [list of (level, parent)]} --> the reason why the value is list of levels is because there may be some topics that appear more than once in the ontology
        inputs:
        > dictionary of ontology parent-child relationship -- the one that has format {parent:[list of children]}
        > dictionary of child-parent relationship -- the one that has format {child:[parents]}
        '''
        returned_dict = {} # format is {topic : [list_of_levels]}
        root = 'computer science'
        returned_dict[root] = [(0, 'ROOT')]
        root_children = parent_children.get(root)
        processed_counter = 0

        unique_nodes = set()
        for k, v in parent_children.items():
            unique_nodes.add(k)
            for item in v:
                unique_nodes.add(item)
        unique_nodes = len(unique_nodes)

        for child in root_children: 
            returned_dict[child] = [(1, root)]
        while processed_counter <= unique_nodes:
            for k, v in parent_children.items():
                parent = k
                children = v
                if parent not in returned_dict:
                    pass
                else:
                    try:
                        item = returned_dict.get(parent)
                        level_parent, level_children = item[0][0], item[0][0] + 1
                        processed_counter += 1
                    except Exception as e:
                        print(e)
                        exit()
                    for child in children:
                        if child not in returned_dict:
                            returned_dict[child] = [(level_children, parent)]
                        else:
                            if (level_children, parent) in returned_dict[child]:
                                pass
                            else:
                                returned_dict[child] += [(level_children, parent)]
            print('{} out of {} ontology terms processed'.format(processed_counter, unique_nodes))
        print()
        return returned_dict

        def get_neighbors(self, node, hops = 5):
            '''
            get the neighbors of the selected ontology node, based on the number of hops (set to 5 by default)
            returns a list of neighbors of the selected node
            '''
            neighbors = []

            return neighbors

class OntologyPrepper():
    def __init__(self,
                 ontology_graph,
                 arguments,
                 ):
        self.ontology_nodes = list(ontology_graph.nodes())
        self.max_topic_len = max([len(topic.split()) for topic in self.ontology_nodes])
        self.node_depth = nx.shortest_path_length(ontology_graph, arguments.onto_graph_root)
        self.max_depth = max(set(self.node_depth.values()))
        if arguments.onto_parents_level != 0:
            self.parents_level = arguments.onto_parents_level - 1
        else:
            raise ValueError('Ontology level have to be more than 0 and less than {}!'.format(self.max_depth))
        self.level_nodes = [] 
        for k, v in self.node_depth.items():
            if v < arguments.onto_parents_level:
                self.level_nodes.append(k)
            elif v == arguments.onto_parents_level:
                self.level_nodes.append(k)
            elif v > arguments.onto_parents_level:
                break
            else:
                pass
        nodes_used = set()
        self.level_nodes_descendants = {} 
        self.descendant_and_parents = {} 
        for curr_cs_topic in self.level_nodes:
            descendants = nx.descendants(ontology_graph, curr_cs_topic)
            self.level_nodes_descendants[curr_cs_topic] = list(descendants)
        for k, v in self.level_nodes_descendants.items():
            for item in v:
                nodes_used.add(item)
                nodes_used.add(k)
                if item not in self.descendant_and_parents:
                    self.descendant_and_parents[item] = [k]
                else:
                    self.descendant_and_parents[item] += [k]
        print('\nThere are {} ontology terms used\n'.format(len(nodes_used)))
        del nodes_used

class OntoKeywordsExtractor():
    '''
    This class is used for:
    > Obtaining ontology keywords that exist in paper abstracts
    > Getting ontology term representation for each paper --> using Node2Vec
    '''
    def __init__(self,
                 max_topic_len,
                 ontology_unique_terms, # ontology_nodes from OntologyPrepper
                 ontology_graph,
                 abstract_embeddings,
                 arguments,
                 descendant_and_parents = None, # from OntologyPrepper
                 paper_keywords = None,
                 paper_abstract = None,
                 use_parent_rep = True, 
                 simple_onto_extend = False,
                 ):

        self.max_topic_len = max_topic_len
        self.ontology_unique_terms = ontology_unique_terms
        self.descendant_and_parents = descendant_and_parents
        use_skipgram = arguments.skipgram
        if use_skipgram:
            skip_distance_range = arguments.skip_distance_range
        else:
            pass

        if paper_keywords == None and paper_abstract == None:
            self.paper_keywords = json.load(open('../data/paper_corresponding_topics_yake.json'))
            self.paper_abstract = json.load(open('../data/paper_paper_abstract_title_bert.json'))
        else:
            self.paper_keywords = paper_keywords
            self.paper_abstract = paper_abstract

        self.paper_windowed_text = {}
        for k, v in tqdm(self.paper_abstract.items()):
            paperID = k
            paper_abstract = self.paper_abstract.get(paperID)[0]
            paper_abstract = paper_abstract.replace(',', '')
            paper_abstract = paper_abstract.replace(':', '')
            paper_abstract = paper_abstract.replace("'", '')
            paper_abstract_sentences = tokenize.sent_tokenize(paper_abstract)
            paper_abstract = self.clean_text(paper_abstract)
            paper_keywords = self.paper_keywords.get(paperID)
            list_of_n = list(range(1, self.max_topic_len + 1))
            paper_abstract_windowed = set()
            for n in list_of_n:
                temp = [' '.join(x) for x in self.window(paper_abstract, n)]
                paper_abstract_windowed.update(temp)
                if use_skipgram: 
                    dist = skip_distance_range
                    if n <= dist:
                        pass
                    else:
                        for sentence in paper_abstract_sentences:
                            sentence = sentence.replace('.', '')
                            sent = sentence.split()
                            skipgram_result = [' '.join(x) for x in list(skipgrams(sent, n, dist))]
                            paper_abstract_windowed.update(skipgram_result)
                else:
                    pass
            self.paper_windowed_text[paperID] = paper_abstract_windowed
            del paper_abstract_windowed
            del paper_keywords
            del paper_abstract_sentences
        self.onto_original_to_plural = {}
        for term in self.ontology_unique_terms:
            pluralized = self.pluralize(term)
            self.onto_original_to_plural[term] = pluralized

        self.paper_terms_in_ontology = {}
        for paperID, windowed_abstract in tqdm(self.paper_windowed_text.items()):
            terms_in_onto = set()
            for k, v in self.onto_original_to_plural.items():
                if v in windowed_abstract:
                    terms_in_onto.add(k)
                else:
                    pass
                if k in windowed_abstract:
                    terms_in_onto.add(k)
                else:
                    pass
            self.paper_terms_in_ontology[paperID] = list(terms_in_onto)

        if simple_onto_extend:
            pretrained_node2vec_path = '../data/pretrained_node2vec_extended.pkl'
        else:
            pretrained_node2vec_path = '../data/pretrained_node2vec.pkl'

        if os.path.isfile(pretrained_node2vec_path):
            print('Pretrained Node2Vec model exists, loading the model...')
            self.node2vec_model = Word2Vec.load(pretrained_node2vec_path)
            print('Pretrained Node2Vec model loaded.\n')
        else:
            print('Pretrained Node2Vec model does not exist, training Node2Vec model...')
            node2vec = Node2Vec(ontology_graph, dimensions = arguments.in_size, walk_length = 30, num_walks = 100, workers = 5)
            self.node2vec_model = node2vec.fit(window = 15, min_count = 1)
            print('Saving Node2Vec model...\n')
            self.node2vec_model.save(pretrained_node2vec_path)

        no_onto_term_counter = 0
        paper_onto_rep_path = 'cache/paper_onto_rep.pkl'
        paper_onto_parent_path = 'cache/paper_onto_parent.json'
        if os.path.isfile(paper_onto_parent_path) and os.path.isfile(paper_onto_rep_path):
            print('Loading ontology parent information cache...')
            print('Part 1...')
            with open(paper_onto_rep_path, 'rb') as f:
                self.paper_onto_rep = pickle.load(f)
            print('Part 2...')
            self.paper_onto_parent = json.load(open(paper_onto_parent_path))
            print('Loaded!\n')
        else:
            print('Creating parent information cache...')
            self.paper_onto_rep = {} 
            self.paper_onto_parent = {}
            for k, v in tqdm(self.paper_terms_in_ontology.items()):
                if len(v) > 0:
                    ontology_terms_representation = []
                    all_onto_parents = set()
                    for onto_term in v:
                        onto_term_parents = self.descendant_and_parents.get(onto_term)
                        try:
                            for parent in onto_term_parents:
                                all_onto_parents.add(parent)
                        except TypeError:
                            pass
                        if not use_parent_rep:
                            onto_term_emb = self.node2vec_model.wv[onto_term]
                        else:
                            if onto_term_parents == None:
                                onto_term_parents = [onto_term]
                            if len(onto_term_parents) > 1:
                                onto_term_emb = []
                                for parent in onto_term_parents:
                                    onto_term_emb.append(self.node2vec_model.wv[parent])
                                onto_term_emb = np.mean(onto_term_emb, axis = 0)
                            else:
                                onto_term_emb = self.node2vec_model.wv[onto_term_parents[0]]
                        ontology_terms_representation.append(onto_term_emb)
                    ontology_terms_representation = np.mean(ontology_terms_representation, axis = 0)
                    self.paper_onto_rep[k] = ontology_terms_representation
                    self.paper_onto_parent[k] = list(all_onto_parents)
                else:
                    no_onto_term_counter += 1
                    pass
            with open(paper_onto_rep_path, 'wb') as handle:
                pickle.dump(self.paper_onto_rep, handle, protocol = pickle.HIGHEST_PROTOCOL)
            json.dump(self.paper_onto_parent, open(paper_onto_parent_path, 'w'))
            print('{} out of {} papers do not have ontology terms'.format(no_onto_term_counter, len(self.paper_onto_rep)))
            print('Paper onto rep and Paper onto parent cache created...\n')

    def clean_text(self, text):
        '''
        Cleaning texts from tanda baca
        '''
        text = text.replace('.', ' ')
        text = text.replace(',', ' ')
        text = text.replace(':', '')
        return text

    def window(self, text, n = 2):
        '''
        Returns a sliding window (of width n) over data from the iterable
        source of code = https://stackoverflow.com/questions/7636004/python-split-string-in-moving-window

        Example:
        input = "7316717"
        output = ["731", "316", "167", "671", "717"]
        '''
        text = [word for word in text.split(' ')]
        it = iter(text)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    def pluralize(self, term):
        words = term.split(' ')
        noun = words[-1]
        if not noun.endswith('s'):
            if re.search('[sxz]$', noun):
                 noun = re.sub('$', 'es', noun)
            elif re.search('[^aeioudgkprt]h$', noun):
                noun = re.sub('$', 'es', noun)
            elif re.search('[aeiou]y$', noun):
                noun = re.sub('y$', 'ies', noun)
            elif noun.endswith('y'):
                noun = noun[:-1]
                noun += 'ies'
            else:
                noun = noun + 's'
        else:
            pass
        words[-1] = noun
        return ' '.join(words)

class UpdateWithHopsInfo():
    '''
    This whole class is used to update the text embedding with additional information of hops from the root topic to each of the current text's parent topics
    '''
    def __init__(self,
                 arguments,
                 paper_terms_in_ontology,
                 abstract_embeddings,
                 descendant_and_parents,
                 node2vec_model,
                 onto_parent_hops,
                 onto_node_node_hops, 
                 mode = 0, # or 1, this is for the term_scoring formula, either dividing by 2 to the power of Hops or just Hops
                 ):

        self.new_abstract_embeddings = {}
        for paper_id, terms in paper_terms_in_ontology.items():
            paper_embed = abstract_embeddings.get(paper_id)
            if len(terms) != 0:
                for term in terms:
                    parents_term = descendant_and_parents.get(term)
                    if parents_term != None: 
                        for parent_term in parents_term:
                            term_n2v_rep = list(node2vec_model.wv[parent_term])
                            curr_term_counter = [t for t in terms if t == term]
                            freq_ratio = len(curr_term_counter) / len(terms) 
                            term_hop = onto_parent_hops.get(parent_term) 
                            if mode == 0:
                                term_score = freq_ratio / pow(2, term_hop) 
                            elif mode == 1:
                                if term_hop == 0:
                                    term_score = 0
                                else:
                                    term_score = freq_ratio / term_hop
                            elif mode == 2 or mode == 3 or mode == 4:
                                n1 = parent_term
                                total_pairs = 0
                                all_distance_scores = []
                                for parent2 in parents_term:
                                    n2 = term
                                    pair = (n1, n2)
                                    pair_name = '{}'.format(pair)
                                    the_hops = onto_node_node_hops.get(pair_name)
                                    if isinstance(the_hops, float):
                                        print('the_hops is float', the_hops)
                                        input()
                                    elif the_hops == None:
                                        print(pair_name)
                                        exit()
                                    else:
                                        hops_n1_n2 = len(the_hops) - 1 
                                    depth_n1 = onto_parent_hops.get(n1)
                                    depth_n2 = onto_parent_hops.get(n2)
                                    D_n1_n2 = hops_n1_n2 / (2 * max(depth_n1, depth_n2))
                                    all_distance_scores.append(D_n1_n2)
                                    total_pairs += 1
                                if mode == 2: 
                                    sum_distance_score = sum(all_distance_scores)
                                    term_score = freq_ratio * sum_distance_score
                                elif mode == 3: 
                                    average_distance_score = sum(all_distance_scores) / len(all_distance_scores)
                                    term_score = freq_ratio * average_distance_score
                                elif mode == 4: 
                                    max_distance_score = max(all_distance_scores)
                                    term_score = freq_ratio * max_distance_score
                            term_rep = [elmnt * term_score for elmnt in term_n2v_rep]
                            paper_embed = [term_rep[i] + paper_embed[i] for i in range(len(paper_embed))]
                    else:
                        term_rep = [0] * arguments.in_size
                        paper_embed = [term_rep[i] + paper_embed[i] for i in range(len(paper_embed))]
            else:
                paper_embed = paper_embed
            self.new_abstract_embeddings[paper_id] = paper_embed

class UpdateWithCoordinatesDistanceInfo():
    '''
    This whole class is used to update the next embedding with additional information of the coordinate distances of each child topic of a current paper with each of the parent topics used in the data
    '''
    def __init__(self,
                 arguments, # args
                 paper_terms_in_ontology,
                 abstract_embeddings,
                 descendant_and_parents,
                 node2vec_model,
                 ):

        self.new_abstract_embeddings = {}
        self.arguments = arguments
        self.embedding_dimension = self.arguments.in_size
        dimension_reductor = self.arguments.dimension_reductor
        self.parent_to_id = {}

        all_parents = set()
        all_parents.add(arguments.onto_graph_root)
        for item in descendant_and_parents.values():
            for x in item:
                all_parents.add(x)
        all_parents = list(all_parents)
        parent_counter = 0
        for parent in all_parents:
            self.parent_to_id[parent] = parent_counter
            parent_counter += 1

        all_parents_rep = []
        all_parents_rep_dict = {}
        for parent in all_parents:
            curr_vec = node2vec_model.wv[parent]
            all_parents_rep.append(curr_vec)
            all_parents_rep_dict[parent] = curr_vec

        coordinate_distance_rep = self.get_coordinate_distance_rep(all_parents_rep_dict)

        assert len(coordinate_distance_rep) == len(all_parents)

        for paper_id, terms in tqdm(paper_terms_in_ontology.items()):
            paper_embed = abstract_embeddings.get(paper_id)
            if len(terms) != 0 and terms != None:
                for term in terms:
                    parents_term = descendant_and_parents.get(term)
                    if parents_term != None:
                        for parent_term in parents_term:
                            parent_rep = coordinate_distance_rep.get(parent_term)
                            curr_term_counter = [t for t in terms if t == term]
                            freq_ratio = len(curr_term_counter) / len(terms)
                            term_rep = [elmnt * freq_ratio for elmnt in parent_rep]
                            paper_embed = [term_rep[i] + paper_embed[i] for i in range(len(paper_embed))]
                    else:
                        term_rep = [0] * arguments.in_size 
                        paper_embed = [term_rep[i] + paper_embed[i] for i in range(len(paper_embed))]
            else:
                paper_embed = paper_embed
            self.new_abstract_embeddings[paper_id] = paper_embed

    def get_coordinate_distance_rep(self, coordinate_dict):
        '''
        This function is for getting the coordinate distance of each parent in the coordinate dict to each other (creating all possible pairs), then resizing it to the desired dimension (args.in_size)
        Outputs a corrdinate distance represenatation dict (coordinate_distance_rep)
        '''
        coordinate_distance_rep_temp = {} 
        for key1 in coordinate_dict.keys():
            coordinate1 = coordinate_dict.get(key1)
            coordinate_distances = [0] * len(coordinate_dict) 
            for key2 in coordinate_dict.keys():
                if key1 != key2:
                    coordinate2 = coordinate_dict.get(key2)
                    if self.arguments.distance_formula == 'euclidean':
                        calculated_dist = distance.euclidean(coordinate1, coordinate2)
                    elif self.arguments.distance_formula == 'inner_product':
                        calculated_dist = np.inner(coordinate1, coordinate2)
                    coordinate_distances[self.parent_to_id.get(key2)] = calculated_dist
            coordinate_distance_rep_temp[key1] = coordinate_distances

        assert len(coordinate_distance_rep_temp) == len(coordinate_dict)

        coordinate_distance_rep = {}
        if self.arguments.dimension_reductor == 'pca':
            reductor = PCA(self.arguments.in_size)
        elif self.arguments.dimension_reductor == 'tsne':
            reductor = TSNE(n_components = self.arguments.in_size, learning_rate = 'auto', init = 'pca', perplexity = 3)

        all_representations = []
        all_keys = []
        for k, v in coordinate_distance_rep_temp.items():
            all_representations.append(v)
            all_keys.append(k)

        resized_representations = reductor.fit_transform(np.array(all_representations))

        for i in range(len(all_keys)):
            key, value = all_keys[i], resized_representations[i]
            coordinate_distance_rep[key] = list(value)

        return coordinate_distance_rep