import torch
import torch.nn.functional as F
from torchmetrics.functional import hamming_distance
import numpy as np
import math
from data_preprocess_clustering import DataPreprocess
from models import RecModelDGI, RecModelLite
from data_loader import InitLoader
from args import *
import networkx as nx
import json, csv, random, datetime, os
from tqdm import tqdm
from time import time
from eval_metrics import mrr, prc, rec, acc, avg, hamming_score
from utils import batch_sampling, batch_sampling_cf, get_node_embeddings, OntologyPrepper, OntoKeywordsExtractor, UpdateWithHopsInfo, UpdateWithCoordinatesDistanceInfo, NeighborsInclusion, huggingface_feature_extrator, OntologyBuilder, read_json_file, NLLLoss
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, hamming_loss, ndcg_score
from fastbm25 import fastbm25
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from copy import deepcopy
import logging
import gc
import ijson

print('\nPytorch version', torch.__version__)

args = make_args()
print('\n', args, '\n')

if args.auto_detect_device:
    if torch.cuda.is_available(): # Use CUDA if available
        print('GPU detected...')
        device_mode = 'cuda'
    else:
        try:
            if torch.backends.mps.is_available():
                print('MAC GPU detected...')
                device_mode = torch.device('mps')
            else:
                print('No MAC GPU detected...')
                device_mode = 'cpu'
        except:
            print('No GPU detected...')
            device_mode = 'cpu'
else:
    device_mode = 'cpu' 

print('Loading paper keywords and abstracts...\n')
paper_keywords = json.load(open('../data/paper_corresponding_topics_yake.json'))
paper_abstract = json.load(open('../data/paper_paper_abstract_title_bert.json'))

try:
    paper_and_year = json.load(open('../data/paper_and_year.json'))
    print('Loaded paper and year information...\n')
except:
    print('Paper and year information does not exist. Creating one now...')
    paper_and_year = {} # dict of paper and the year it is published, CHECKPOINT
    if args.dynamic_onto_extend:
        paper_info = json.load(open('../data/sessions.json')) # including paper title, paper id, authors, cited papers per sections, etc
        dblp_paper_info = json.load(open('../data/dblp_abstract_keywords.json')) # more information about each paper including paper title, paper id, year, etc
        temp = json.load(open('../data/paper_to_title.json'))
        paperid_to_title, title_to_paperid = {}, {}
        for k, v in temp.items():
            if type(v) == list:
                paperid_to_title[k] = v[0]
                title_to_paperid[v[0]] = k
            else:
                paperid_to_title[k] = v
                title_to_paperid[v] = k

        for k, v in paper_info.items():
            paper_id = ''
            paper_year = None
            for key, val in v.items():
                if key == 'paper_id':
                    paper_id = val
                elif key == 'year':
                    paper_year = int(val)
                else:
                    pass
            paper_and_year[paper_id] = paper_year

        for k, v in dblp_paper_info.items():
            paper_title = ''
            paper_year = None
            for key, val in v.items():
                if key == 'ori_title':
                    paper_title = val
                elif key == 'year':
                    paper_year = int(val)
            try:
                paper_id = title_to_paperid.get(paper_title)
            except Exception as e:
                print(e)
                exit()
            paper_and_year[paper_id] = paper_year
        filename = open('../data/paper_and_year.json', 'w')
        json.dump(paper_and_year, filename)
        print('Paper and year information created...\n')

# loading the necessary files
if args.remove_sparse:
    filename = '../data_preprocessed/sessions.json'
    paper_info = json.load(open(filename))
    filename = '../data_preprocessed/authors_cite_authors.json'
    global_authors = json.load(open(filename))
elif not args.remove_sparse:
    filename = '../data/sessions.json'
    paper_info = json.load(open(filename))
    filename = '../data/authors_cite_authors.json'
    global_authors = json.load(open(filename))
    for k, v in global_authors.items(): 
        temp = []
        for item in v:
            if item[0] == None:
                pass
            else:
                temp.append(item)
        global_authors[k] = temp

if args.use_abstract_embed:
    print('Using {} abstract embeddings\n'.format(args.embedding_mode))
    if args.embedding_mode == 'glove':
        filename = '../data/abstract_embedding_glove.json'
    elif args.embedding_mode == 'bert':
        filename = '../data/abstract_embedding_bert.json' # each embedding consists of title, abstract, and keywords embeddings. Format is {worded_paper_ID : embedding}
    elif args.embedding_mode == 'glove_incomplete': # glove without keywords
        filename = '../data/abstract_embedding.json' # change later, delete the 'copy'
    elif args.embedding_mode == 'llm_small':
        filename = 'cache/LLM_model_text-embedding-small-3.json'
    abstract_embeddings = json.load(open(filename))
else:
    abstract_embeddings = None

if args.use_author_embed: # this version of author embedding is still partial ---> meaning not all authors have good keywords representation
    print('Using {} author embeddings\n'.format(args.embedding_mode))
    if args.embedding_mode == 'glove':
        filename = '../data/author_embedding_bert.json' # glove embedding has not been made yet, for now using bert embedding even for 'glove' mode
    elif args.embedding_mode == 'bert':
        filename = '../data/author_embedding_bert.json'
    elif args.embedding_mode == 'llm_small':
        filename = 'cache/LLM_model_text-embedding-small-3_author_embed.json'
    author_embeddings = json.load(open(filename))
else:
    pass

if args.use_hftransformer: # loading huggingface paper embedding cache
    print('Handling Hugginface embedding...\n')
    if args.hf_model == 'gpt2':
        filename = 'cache/paper_embedding_gpt2.json'
    else:
        pass
    try: # try loading a paper cache file
        hf_embed_temp = json.load(open(filename))
    except: # cache file does not exist, making one
        hf_embed_temp = huggingface_feature_extrator(paper_abstract, dimentionality_reductor = args.hf_dimentionality_reductor, target_dimension = args.in_size, hf_model = args.hf_model)
        json.dump(hf_embed_temp, open(filename, 'w')) # uncomment to save the file later TO DO
    hf_embed = {}
    for k, v in hf_embed_temp.items():
        hf_embed[k] = torch.FloatTensor(v)

ontology_rep_embedding = None
descendant_and_parents = None
paper_onto_parent = None

# this is for creating ontology, used in the dynamic ontology extension run too so I place this outside of the IF condition
print('Building ontology...\n')
ob = OntologyBuilder(paper_abstract, paper_and_year, args.ontox_classifier, args.ontox_status, args.dynamic_onto_year, device_mode, train_year_range = args.year_range, dynamic_onto = args.dynamic_onto, vanilla_cso = args.vanilla_cso, onto_important_terms_only = args.onto_important_terms_only, term_importance = args.term_importance, document_lemmatization = args.document_lemmatization, bm25_representative_measure = args.bm25_representative_measure, paper_keywords = paper_keywords, simple_onto_extend = args.simple_onto_extend, similarity_for_onto_extend = True, dynamic_onto_extend = args.dynamic_onto_extend, triple_types_experiment = args.triple_types_experiment)

if args.coordinate_distance or args.onto_hops or args.kw_neighbors_in_onto: # if use onto hops
    ontology_edges = ob.ontology_edges # dict of edges, format is {source_node : [list of target_nodes]}
    ontology_parent_children, ontology_child_parents = ob.ontology_parent_children, ob.ontology_child_parents
    ontology_node_levels = ob.ontology_node_levels
    unimportant_terms = ob.unimportant_terms
    ontology_graph = ob.ontology_graph
    ontology_directed_graph = ob.ontology_graph_directed
    topic_matches = ob.topic_matches # these are the topic matches of each keyword given a paper

    # updating the keywords embedding to also contain the emebddings of its neighbors
    if args.kw_neighbors_in_onto:
        # make additional keywords information which include the neighbors of each paper keyword
        print('Getting keyword neighbors information from the ontology...')
        distance = args.kw_neighbor_distance # radius for ego graph
        cache_filename = 'cache/paper_keywords_neighbors_with_distance_{}.json'.format(distance)
        try: # loading neighbor cache
            with open(cache_filename) as f:
                paper_keywords_neighbors = json.load(f)
            # paper_keywords_neighbors = json.load(open(cache_filename))
        except:
            print('Creating paper keywords neighbors cache for radius = {}...'.format(distance))
            paper_keywords_neighbors = {} # {keyword : [list of keyword's neighbor in ontology, including this keyword itself]}
            for k, v in tqdm(paper_keywords.items()):
                for curr_kw in v:
                    try:
                        ego_graph_for_curr_kw = nx.generators.ego_graph(ontology_directed_graph, curr_kw, distance)
                        curr_kw_neighbors = list(ego_graph_for_curr_kw.nodes())
                        paper_keywords_neighbors[curr_kw] = curr_kw_neighbors
                    except: # keyword does not exist inside of the ontology
                        paper_keywords_neighbors[curr_kw] = [curr_kw]
            # dump paper_keywords_neighbors dict as json file cache
            print('Saving cache...')
            json.dump(paper_keywords_neighbors, open(cache_filename, 'w'))
            print('Cache saved!\n')
        # get the extended keywords embeddings start here
        ni = NeighborsInclusion(paper_keywords, paper_keywords_neighbors, distance = distance)

    print('Preparing ontology...')
    op = OntologyPrepper(ontology_directed_graph, args) # located in utils.py
    print('Ontology prepared...\n')
    print('Extracting ontology keywords...')
    oke = OntoKeywordsExtractor(op.max_topic_len, op.ontology_nodes, ontology_directed_graph, abstract_embeddings, args, descendant_and_parents = op.descendant_and_parents, paper_keywords = paper_keywords, paper_abstract = paper_abstract, simple_onto_extend = args.simple_onto_extend) # located in utils.py
    print('Ontology Keywords extracted...\n')
    level_nodes_descendants = op.level_nodes_descendants
    ontology_rep_embedding = oke.paper_onto_rep
    descendant_and_parents = op.descendant_and_parents
    paper_onto_parent = oke.paper_onto_parent
    no_path_counter = 0
    all_pairs_counter = 0

    try:
        print('Loading hops cache...')
        parser = ijson.parse(open('cache/onto_parent_hops.json'))
        onto_parent_hops = {}
        for k, v in tqdm(ijson.kvitems(open('cache/onto_parent_hops.json'), '')):
            onto_parent_hops[k] = v
        # onto_parent_hops = json.load(open('cache/onto_parent_hops.json')) # format is {onto_parent : onto_parent_hop_to_root} A hop is len(shortest_path)-1
        print('Onto parent hops cache loaded')
        # onto_node_node_hops = read_json_file('cache/onto_node_node_hops.json') # format is {'(onto_parent1, onto_parent2)': hop value from onto_parent1 to onto_parent2}
        # print('Onto node node hops loaded')
        # print('Loaded!\n')
    except Exception as e:
        print(e)
        print('Creating hops information between parent (category) node pairs...')
        onto_parent_hops = {} # format is {onto_parent : onto_parent_hop_to_root} A hop is len(shortest_path)-1
        onto_node_node_hops = {} # format is {'(onto_parent1, onto_parent2)': hop value from onto_parent1 to onto_parent2}
        for k, v in tqdm(level_nodes_descendants.items()):
            try:
                shortest_path_to_root = nx.shortest_path(ontology_graph, source = args.onto_graph_root, target = k)
                onto_parent_hops[k] = len(shortest_path_to_root) - 1
            except:
                print(k, 'no path')
                onto_parent_hops[k] = math.inf # no path existing betwen the current node and computer science, if this is the case then the hops value would be infinity
            # make all possible pairs of the parent nodes
            all_parent_nodes = list(level_nodes_descendants.keys())
            for parent_node in all_parent_nodes:
                try:
                    shortest_path_between_nodes = nx.shortest_path(ontology_graph, source = k, target = parent_node)
                    pair = (k, parent_node)
                    onto_node_node_hops['{}'.format(pair)] = shortest_path_between_nodes
                except:
                    try:
                        shortest_path_between_nodes = nx.shortest_path(ontology_graph, source = parent_node, target = k)
                        pair = (k, parent_node) # pair tetap sama, asumsi the hops work both ways, i.e., parent node to k OR k to parent node
                        onto_node_node_hops['{}'.format(pair)] = shortest_path_between_nodes
                    except:
                        # print('{} and {} have no path'.format(k, parent_node))
                        onto_node_node_hops['{}'.format(pair)] = math.inf # set the nodes that have no path as infinity
                        no_path_counter += 1
                all_pairs_counter += 1
        json.dump(onto_parent_hops, open('cache/onto_parent_hops.json', 'w'))
        json.dump(onto_node_node_hops, open('cache/onto_node_node_hops.json', 'w'))
        print('{} out of all {} pairs have no path in the ontology graph\n'.format(no_path_counter, all_pairs_counter))

    try: # if cache hops embedding exists
        if args.onto_hops:
            print('Loading topic hops information from the ontology...\n')
            hops_emb_filename = open('run_results/abstract_embeddings_with_hops_{}_levels_{}.json'.format(args.onto_parents_level, 'onto_hops'))
        elif args.coordinate_distance:
            print('Loading topic coordinates distance representation...\n')
            hops_emb_filename = open('run_results/abstract_embeddings_with_hops_{}_levels_{}.json'.format(args.onto_parents_level, 'coordinate_dist'))
    except: # if the cache hops embedding does not exist, then make one
        print('Combining text embedding with terms\' hops/coordinate representation...') # TO DO: save this into a json file
        if args.onto_hops:
            print('Obtaining topic hops information from the ontology...\n')
            updater = UpdateWithHopsInfo(args, oke.paper_terms_in_ontology, abstract_embeddings, op.descendant_and_parents, oke.node2vec_model, onto_parent_hops, onto_node_node_hops, mode = args.term_score_mode) # hops info updater --> used to update the text embedding with hops information
            hops_emb_filename = open('run_results/abstract_embeddings_with_hops_{}_levels_{}.json'.format(args.onto_parents_level, 'onto_hops'), 'w')
        elif args.coordinate_distance:
            print('Obtaining topic coordinates distance representation...\n')
            updater = UpdateWithCoordinatesDistanceInfo(args, oke.paper_terms_in_ontology, abstract_embeddings, op.descendant_and_parents, oke.node2vec_model) # coordinate distance info updater --> used to update the text embedding with coordinate distance info
            hops_emb_filename = open('run_results/abstract_embeddings_with_hops_{}_levels_{}_{}.json'.format(args.onto_parents_level, 'coordinate_dist', args.distance_formula), 'w')
        abstract_embeddings = updater.new_abstract_embeddings
        json.dump(abstract_embeddings, hops_emb_filename) # save the json file for future use

    if args.kw_neighbors_in_onto:
        print('Including keyword neighbors information...\n')
        abstract_embedding = json.load(hops_emb_filename)
        abstract_embedding = ni.include_kw_neighbors_embeddings(abstract_embedding)
    else:
        abstract_embeddings = json.load(hops_emb_filename)

else:
    ontology_edges = None # set to None to represent that the ontology has not been loaded and processed

if args.clustering_method == 'ontology' or args.clustering_method == 'vec2gc':
    if ontology_edges == None:
        print('Preparing ontology...')
        csv_file = open('../data/onto_dictionary_nolimit.csv') # csv format is (child, parent)
        ontology_edges = {} # dict of edges, format is {source_node : [list of target_nodes]}
        ontology_parent_children, ontology_child_parents = {}, {}
        ontology_node_levels = {}
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            child, parent = row[0].replace('_', ' '), row[-1].replace('_', ' ')
            if parent not in ontology_edges:
                ontology_edges[parent] = [child]
            else:
                ontology_edges[parent] += [child]

            if parent not in ontology_parent_children:
                ontology_parent_children[parent] = [child]
            else:
                ontology_parent_children[parent] += [child]

            if child not in ontology_child_parents:
                ontology_child_parents[child] = [parent]
            else:
                ontology_child_parents[child] += [parent]
        ontology_graph = nx.DiGraph(ontology_edges)
        # ontology_graph = nx.Graph(ontology_edges) # set to non directed graph to get hops
        op = OntologyPrepper(ontology_graph, args) # located in utils.py
        print('Ontology prepared...')
        oke = OntoKeywordsExtractor(op.max_topic_len, op.ontology_nodes, ontology_graph, abstract_embeddings, args, descendant_and_parents = op.descendant_and_parents, paper_keywords = paper_keywords, paper_abstract = paper_abstract) # located in utils.py
        print('Ontology Keywords extracted...\n')
        # level_nodes_descendants = op.level_nodes_descendants
        ontology_rep_embedding = oke.paper_onto_rep
        descendant_and_parents = op.descendant_and_parents
        paper_onto_parent = oke.paper_onto_parent
    else:
        print('Ontology has been prepared beforehand, do not need further process...\n')
        pass
del ontology_edges
gc.collect()

if args.dynamic_onto:
    test_years = [args.dynamic_onto_year]
else:
    test_years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    test_years = [2017, 2018, 2019, 2020]

if len(test_years) == 1:
    print('\nTest year used is only {}\n'.format(test_years[-1]))
else:
    print('\nTest years from {} to {}\n'.format(test_years[0], test_years[-1]))

clusters_based_on_year = {} # this is for the clustering, the value is based on the elbow method and gap statistics result which can be obtained from declaring --elbow_method OR --gap_statistics into the command prompt bash. Format is {test_year : num_cluster}
for test_year in test_years:
    clusters_based_on_year[test_year] = args.num_clusters

neighbor_size = [15, 10, 5]
all_k = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
similarity_threshold = args.sim_threshold
cur_time = ''.join([char for char in str(datetime.datetime.now()) if char != '-' and char != ':' and char != ' ' and char != '.'])

run_results_dir = 'run_results/ep{}_iomode{}_b{}_lr{}_{}'.format(args.epoch, args.input_output_format, args.batch_size, args.learning_rate, cur_time).replace('.', '')
saved_model_dir = 'saved_model/ep{}_iomode{}_b{}_lr{}_{}'.format(args.epoch, args.input_output_format, args.batch_size, args.learning_rate, cur_time).replace('.', '')

print('\nRun results directory: {}\n'.format(run_results_dir))

os.mkdir(run_results_dir)
os.mkdir(saved_model_dir)

all_predicted_clusters = {} # this is for the retrieval step, stores predicted results in the format of {test_year : [list of clusters predicted by the classifier]}
all_inputs = {} # this is for the retrieval step, stores predicted results in the format of {test_year : [list of input papers]}
all_y_true_multilabel = {} # this is for the retrieval step, stores the worded gold multilabels in the format of {test_year : y_true_multilabel}
all_papers_wrt_cluster = {} # this is for the retrieval step, format is {test_year : papers_wrt_cluster}

il = InitLoader()

for test_year in test_years:

    print('\nStart {} process with year {} as test data'.format(device_mode.upper(), test_year))

    path = '{}/RecModel_{}_{}_{}.pth'.format(saved_model_dir, args.nn_model, test_year, cur_time) # this is the save filename

    dp = DataPreprocess(paper_info,
                        global_authors,
                        test_year,
                        clusters_based_on_year,
                        il.complete_coauth_features,
                        il.complete_citation_features,
                        il.complete_authorship_features,
                        il.complete_section_features,
                        abstract_embeddings = abstract_embeddings,
                        paper_abstract = paper_abstract,
                        author_embeddings = author_embeddings,
                        ontology_rep_embedding = ontology_rep_embedding,
                        paper_and_onto_parent = paper_onto_parent,
                        )
    print('Finished data preprocessing...\n')

    y_true_multilabel = dp.tes_sections_output_worded_multi # already worded version for testing
    y_true_multilabel_tra = dp.tra_sections_output_worded_multi

    print('Getting the max number of section in a paper in the whole data...')
    if args.section_embedding:
        max_section = 0
        for item in dp.tra_sections_inputs_pos:
            if isinstance(item, list):
                if max_section < max(item):
                    max_section = max(item)
            elif isinstance(item, int):
                if max_section < item:
                    max_section = item
        for item in dp.tes_sections_inputs_pos:
            if isinstance(item, list):
                if max_section < max(item):
                    max_section = max(item)
            elif isinstance(item, int):
                if max_section < item:
                    max_section = item
        print('Max number of section is {}\n'.format(max_section))
    else:
        max_section = None

    if args.lite_run:
        model = RecModelLite(dp.num_classes,
                             device_mode,
                             similarity_feat = dp.similarity_features,
                             neighbor_sizes = neighbor_size,
                             nn_model = args.nn_model,
                             node_emb_dim = args.in_size,
                             hidden_dim1 = args.hidden_size1,
                             hidden_dim2 = args.hidden_size2,
                             output_dim = args.nn_model_out_size,
                             use_dgi = args.use_dgi,
                             GRU = args.use_gru,
                             lite_GRU = args.lite_gru,
                             GRU_MHA = args.use_gru_mha,
                             MHA = args.use_mha,
                             position_encoding = args.use_position_encoding,
                             max_seq_len = max_section,
                             idx_to_ppr = dp.idx_to_ppr,
                             idx_to_author = dp.idx_to_author,
                             ).to(device_mode)

    elif args.lite_run and args.lite_gru:
        pass

    elif not args.lite_run and not args.lite_gru:
        model = RecModelDGI(dp.num_classes,
                            device_mode,
                            similarity_feat = dp.similarity_features,
                            neighbor_sizes = neighbor_size,
                            nn_model = args.nn_model,
                            node_emb_dim = args.in_size,
                            hidden_dim1 = args.hidden_size1,
                            hidden_dim2 = args.hidden_size2,
                            output_dim = args.nn_model_out_size,
                            use_dgi = args.use_dgi,
                            GRU = args.use_gru,
                            GRU_MHA = args.use_gru_mha,
                            MHA = args.use_mha,
                            position_encoding = args.use_position_encoding,
                            max_seq_len = max_section,
                            idx_to_ppr = dp.idx_to_ppr,
                            idx_to_author = dp.idx_to_author,
                            ).to(device_mode)

    ''' TRAINING LOOP '''
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    losses = {}
    epoch_losses = []
    epoch_corrects = []

    print('\n=== Start training using test data from year {} === '.format(test_year))
    print('Number of training instances', len(dp.tra_sections_inputs))
    print("Batch sample size:", args.batch_size)

    all_batches_precision, all_batches_recall, all_batches_acc, all_batches_mrr, all_batches_losses = {}, {}, {}, {}, {}
    all_epochs_precision, all_epochs_recall, all_epochs_acc, all_epochs_mrr, all_epochs_losses = {}, {}, {}, {}, {}

    highest_precision = -math.inf
    for epoch in range(args.epoch):

        start_time = time()

        tra_outputs = dp.tra_sections_outputs_second_index

        if args.input_output_format == 1 or args.input_output_format == 42 or args.input_output_format == 43:
            if args.use_cf:
                assert len(dp.tra_sections_authors) == len(dp.tra_neighborhoods_representations)
                authors_batches, neighbor_rep_batches, inputs_batches, inputs_pos_batches, labels_batches = batch_sampling_cf(args.batch_size,
                                      dp.tra_sections_authors,
                                      dp.tra_neighborhoods_representations,
                                      dp.tra_sections_inputs,
                                      dp.tra_sections_inputs_pos,
                                      tra_outputs,
                                      shuffling = True)
                assert len(authors_batches) == len(neighbor_rep_batches)
                assert len(inputs_batches) == len(inputs_pos_batches)
            else:
                authors_batches, inputs_batches, inputs_pos_batches, labels_batches = batch_sampling(args.batch_size,
                              dp.tra_sections_authors,
                              dp.tra_sections_inputs,
                              dp.tra_sections_inputs_pos,
                              tra_outputs,
                              shuffling = True)
                try:
                    assert len(inputs_batches) == len(inputs_pos_batches)
                except Exception as e:
                    print(e)
                    print(len(inputs_batches), len(inputs_pos_batches))
                    input()
                neighbor_rep_batches = None
        elif args.input_output_format == 2 or args.input_output_format == 3:
            if args.use_cf:
                pass
            else:
                authors_batches, inputs_batches, inputs_pos_batches, labels_batches = batch_sampling(args.batch_size,
                              dp.tra_sections_authors,
                              dp.tra_sections_inputs,
                              dp.tra_sections_inputs_pos,
                              dp.tra_sections_outputs_ohe,
                              shuffling = True)
                assert len(inputs_batches) == len(inputs_pos_batches)
                neighbor_rep_batches = None
        elif args.input_output_format == 22 or args.input_output_format == 32:
            if args.use_cf:
                pass
            else:
                authors_batches, inputs_batches, inputs_pos_batches, labels_batches = batch_sampling(args.batch_size,
                              dp.tra_sections_authors,
                              dp.tra_sections_inputs,
                              dp.tra_sections_inputs_pos,
                              dp.tra_sections_outputs_fo,
                              shuffling = True)
                assert len(inputs_batches) == len(inputs_pos_batches)
                neighbor_rep_batches = None

        acc_loss = 0

        for batch in tqdm(authors_batches):
            assert len(authors_batches) == len(labels_batches)
            b = authors_batches.index(batch) # index of the current batch
            if neighbor_rep_batches != None:
                neighbor_rep_batch = neighbor_rep_batches[b]
            else:
                neighbor_rep_batch = neighbor_rep_batches

            optimizer.zero_grad()

            if args.nn_model == 'sage':
                with torch.enable_grad():
                    prediction = model.forward(authors_batches[b], # author_batch
                                            dp.nd_types, # node_types_dict
                                            dp.num_authors, dp.num_papers, neighbor_size,
                                            # from coauthorship graph (homogeneous)
                                            torch.as_tensor(dp.coauth_features.float(), device = device_mode), # coauth_features, feature matrix utk training dan testing sama
                                            torch.as_tensor(dp.coauth_tra_edge_idx, device = device_mode), # coauth_adj
                                            # from citation graph (homogeneous)
                                            # dp.coauth_n2v_tes,
                                            torch.as_tensor(dp.citation_features.float(), device = device_mode), # citation_features
                                            torch.as_tensor(dp.citation_tra_edge_idx, device = device_mode), # citation_adj
                                            # dp.citation_n2v_tes,
                                            # from authorship graph (heterogeneous)
                                            torch.as_tensor(dp.authorship_features.float(), device = device_mode), # authorship_pca (not actually using pca though) information about the edges
                                            # authorship_indexer.encoders,
                                            torch.as_tensor(dp.authorship_tra_edge_idx, device = device_mode), # authorship_adj
                                            # dp.authorship_n2v_tes,
                                            inputs_batches[b], # section_batch
                                            inputs_pos_batches[b],
                                            torch.as_tensor(dp.section_features.float(), device = device_mode), # section_features
                                            torch.as_tensor(dp.section_tra_edge_idx, device = device_mode),
                                            # dp.section_n2v_tes,
                                            neighbor_rep_batch,
                                            ) # section_adj

                    # if args.input_output_format == 1 or args.input_output_format == 42 or args.input_output_format == 43:
                    pred_softmax = prediction.log_softmax(dim=-1) # use log softmax for the rest
                    # pred_softmax = prediction.softmax(dim=-1) # use softmax for llm version
                    # print(pred_softmax, pred_softmax.shape)
                    try:
                        loss = NLLLoss(pred_softmax, torch.LongTensor(labels_batches[b]))
                        # if args.embedding_mode == 'llm_small':
                        #     loss = NLLLoss(pred_softmax, torch.LongTensor(labels_batches[b])) # using custom NLL loss function
                        # else:
                        #     loss = F.nll_loss(pred_softmax, torch.LongTensor(labels_batches[b]).to(device_mode))
                        acc_loss += loss
                        loss.backward() # backpropagation
                        optimizer.step()
                    except Exception as e: # if there is nan in the softmax result, change it to zeros
                        # this is a tambal sulam solution, because softmax returns nan with our new LLM based text embeddings
                        # print(e)
                        temp = pred_softmax.tolist()
                        final = []
                        for i in range(len(temp)):
                            item = torch.FloatTensor(temp[i])
                            verdict = torch.isnan(item)
                            verdict = verdict.tolist()
                            if verdict[0]:
                                new = [0] * len(verdict)
                                final.append(new)
                            else:
                                final.append(temp[i])
                        pred_softmax = torch.FloatTensor(final)
                        loss = NLLLoss(pred_softmax, torch.LongTensor(labels_batches[b]))
                        acc_loss += loss
                        loss.requires_grad = True
                        loss.backward() # backpropagation
                        optimizer.step()

                if args.test_per_batch:
                    print('Start Eval of Current BATCH')
                    tes_y_true = dp.tes_sections_outputs_second_index # gold label for test data
                    batch_correct_counter = 0
                    batch_prediction_results = []
                    batch_gold = []
                    with torch.no_grad():
                        tes_prediction = model.forward(dp.tes_sections_authors,
                                                            dp.nd_types,
                                                            dp.num_authors, dp.num_papers, neighbor_size,
                                                            torch.Tensor.float(dp.coauth_features),
                                                            torch.Tensor(dp.coauth_tes_edge_idx),
                                                            # dp.coauth_n2v_tes,
                                                            torch.Tensor.float(dp.citation_features),
                                                            torch.Tensor(dp.citation_tes_edge_idx),
                                                            # dp.citation_n2v_tes,
                                                            torch.Tensor.float(dp.authorship_features),
                                                            torch.Tensor(dp.authorship_tes_edge_idx),
                                                            # dp.authorship_n2v_tes,
                                                            dp.tes_sections_inputs, # section_batch
                                                            dp.tes_sections_inputs_pos,
                                                            torch.Tensor.float(dp.section_features),
                                                            torch.Tensor(dp.section_tes_edge_idx),
                                                            # dp.section_n2v_tes,
                                                            dp.tes_neighborhoods_representations,
                                                            )
                        pred_softmax = tes_prediction.softmax(dim=-1)
                        pred_softmax_list = pred_softmax.tolist()
                        pred_log_softmax = tes_prediction.log_softmax(dim=-1)
                        pred_log_softmax = pred_log_softmax.cpu()
                        pred_log_softmax_list = pred_log_softmax.tolist()
                        batch_loss_small = NLLLoss(pred_log_softmax, torch.LongTensor(tes_y_true))
                        print('Batch test loss: {}'.format(batch_loss_small.detach()))

                        for i in range(len(tes_y_true)):
                            try:
                                curr_pred = pred_softmax[i].detach()
                                curr_pred = curr_pred.cpu()
                                curr_pred = curr_pred.numpy()
                                gold_pred = int(tes_y_true[i])
                                batch_gold.append(gold_pred)
                                top_1_pred = int(np.argmax(curr_pred))
                                batch_prediction_results.append(top_1_pred)
                                if top_1_pred == gold_pred:
                                    batch_correct_counter += 1
                            except Exception as e:
                                print('EXCEPTION OCCURRED:', e)
                                pass
                        batch_precision = round(precision_score(batch_gold, batch_prediction_results, average = 'macro', zero_division = 0), 4)
                        batch_recall = round(recall_score(batch_gold, batch_prediction_results, average = 'macro', zero_division = 0), 4)
                        batch_acc = round(accuracy_score(batch_gold, batch_prediction_results), 4)

                        if batch_precision > highest_precision:
                            print('Higher precision found, saving the model midway...')
                            highest_precision = batch_precision
                            torch.save(model.state_dict(), path)
                            print('current highest precision =', highest_precision)
                        else:
                            pass

                        print('Batch Precision: {} || Batch Recall: {} || Batch Accuracy: {}'.format(batch_precision, batch_recall, batch_acc))
                        print('{} correct predictions out of {}\n'.format(batch_correct_counter, len(dp.tes_sections_authors)))

        if args.test_per_epoch:
            print('Start Eval of Current EPOCH')

            tes_y_true = dp.tes_sections_outputs_second_index
            epoch_correct_counter = 0
            epoch_prediction_results = []
            epoch_gold = []

            with torch.no_grad():
                tes_prediction = model.forward(dp.tes_sections_authors,
                                                dp.nd_types,
                                                dp.num_authors, dp.num_papers,  neighbor_size,
                                                torch.Tensor.float(dp.coauth_features),
                                                torch.Tensor(dp.coauth_tes_edge_idx),
                                                # dp.coauth_n2v_tes,
                                                torch.Tensor.float(dp.citation_features),
                                                torch.Tensor(dp.citation_tes_edge_idx),
                                                # dp.citation_n2v_tes,
                                                torch.Tensor.float(dp.authorship_features),
                                                torch.Tensor(dp.authorship_tes_edge_idx),
                                                # dp.authorship_n2v_tes,
                                                dp.tes_sections_inputs, # section batch
                                                dp.tes_sections_inputs_pos,
                                                torch.Tensor.float(dp.section_features),
                                                torch.Tensor(dp.section_tes_edge_idx),
                                                # dp.section_n2v_tes,
                                                dp.tes_neighborhoods_representations,
                                                )
                pred_softmax = tes_prediction.softmax(dim=-1)
                pred_log_softmax = tes_prediction.log_softmax(dim=-1)
                pred_log_softmax = pred_log_softmax.cpu()
                epoch_loss_small = NLLLoss(pred_log_softmax, torch.LongTensor(tes_y_true))
                print('Test data loss: {}'.format(epoch_loss_small.detach()))

                for i in range(len(tes_y_true)):
                    try:
                        curr_pred = pred_softmax[i].detach()
                        curr_pred = curr_pred.cpu()
                        curr_pred = curr_pred.numpy()
                        gold_pred = int(tes_y_true[i])
                        epoch_gold.append(gold_pred)
                        top_1_pred = int(np.argmax(curr_pred))
                        epoch_prediction_results.append(top_1_pred)
                        if top_1_pred == gold_pred:
                            epoch_correct_counter += 1
                    except Exception as e:
                        print('EXCEPTION OCCURRED:', e)
                        pass
                epoch_precision = round(precision_score(epoch_gold, epoch_prediction_results, average = 'macro', zero_division = 0), 4)
                epoch_recall = round(recall_score(epoch_gold, epoch_prediction_results, average = 'macro', zero_division = 0), 4)
                epoch_acc = round(accuracy_score(epoch_gold, epoch_prediction_results), 4)

                if epoch_precision > highest_precision:
                    print('Higher precision found, saving the model midway...')
                    highest_precision = epoch_precision
                    torch.save(model.state_dict(), path)
                    print('current highest precision =', highest_precision)
                else:
                    pass

                print('Epoch Precision: {} || Epoch Recall: {} || Epoch Accuracy: {}'.format(epoch_precision, epoch_recall, epoch_acc))
                print('{} correct predictions out of {}'.format(epoch_correct_counter, len(dp.tes_sections_authors)))

        try:
            epoch_loss = float(acc_loss.item() / len(authors_batches))
        except AttributeError:
            epoch_loss = float(acc_loss / len(authors_batches))

        epoch_losses.append(epoch_loss)
        print('Epoch {} took {} minutes to finish, loss {}\n'.format(epoch + 1, round((time() - start_time)/60, 2), round(epoch_loss, 5)))

        # print("loss of last batch", loss)
        losses[epoch + 1] = round(epoch_loss, 4)

    ''' SAVING MODEL '''
    if args.test_per_batch or args.test_per_epoch:
        pass
    else:
        torch.save(model.state_dict(), path)

    print('Loss average is {}\n'.format(round(sum(losses.values())/len(losses), 4)))

    # TEST LOOP for this fold
    print('=== Start Testing ===')

    print('\nNumber of testing instances', len(dp.tes_sections_inputs))

    test_device = torch.device('cpu')

    # load model
    saved_model_path = '{}/RecModel_{}_{}_{}.pth'.format(saved_model_dir, args.nn_model, test_year, cur_time)
    assert saved_model_path == path

    print('Num classes', dp.num_classes)

    if args.lite_run:
        loaded_model = RecModelLite(dp.num_classes,
                             test_device,
                             similarity_feat = dp.similarity_features,
                             neighbor_sizes = neighbor_size,
                             nn_model = args.nn_model,
                             node_emb_dim = args.in_size,
                             hidden_dim1 = args.hidden_size1,
                             hidden_dim2 = args.hidden_size2,
                             output_dim = args.nn_model_out_size,
                             use_dgi = args.use_dgi,
                             GRU = args.use_gru,
                             lite_GRU = args.lite_gru,
                             GRU_MHA = args.use_gru_mha,
                             MHA = args.use_mha,
                             position_encoding = args.use_position_encoding,
                             max_seq_len = max_section,
                             idx_to_ppr = dp.idx_to_ppr,
                             idx_to_author = dp.idx_to_author,
                             )
    else:
        loaded_model = RecModelDGI(dp.num_classes,
                            test_device,
                            similarity_feat = dp.similarity_features,
                            neighbor_sizes = neighbor_size,
                            nn_model = args.nn_model,
                            node_emb_dim = args.in_size,
                            hidden_dim1 = args.hidden_size1,
                            hidden_dim2 = args.hidden_size2,
                            output_dim = args.nn_model_out_size,
                            use_dgi = args.use_dgi,
                            GRU = args.use_gru,
                            GRU_MHA = args.use_gru_mha,
                            MHA = args.use_mha,
                            position_encoding = args.use_position_encoding,
                            max_seq_len = max_section,
                            idx_to_ppr = dp.idx_to_ppr,
                            idx_to_author = dp.idx_to_author,
                            )

    loaded_model.load_state_dict(torch.load(saved_model_path, map_location = test_device)) # loading the model's parameters
    loaded_model.eval() # to stop dropout and gradietn computation

    # PREDICTION: we do not use batching during prediction, just all data as one batch

    if args.input_output_format == 1 or args.input_output_format == 42 or args.input_output_format == 43:
        if args.clustering_method == 'vec2gc':
            y_true = dp.tes_sections_outputs_second_index
            y_true_tra = dp.tra_sections_outputs_second_index
        else:
            y_true = dp.tes_sections_outputs
            y_true_tra = dp.tra_sections_outputs

    elif args.input_output_format == 2 or args.input_output_format == 3:
        y_true = dp.tes_sections_outputs_ohe

    if args.nn_model == 'sage':
        prediction = loaded_model.forward(dp.tes_sections_authors, # author batch
                                    dp.nd_types,
                                    dp.num_authors, dp.num_papers, neighbor_size,
                                    torch.Tensor.float(dp.coauth_features), # feature matrix utk training dan testing sama
                                    torch.Tensor(dp.coauth_tes_edge_idx),
                                    torch.Tensor.float(dp.citation_features),
                                    torch.Tensor(dp.citation_tes_edge_idx),
                                    torch.Tensor.float(dp.authorship_features), # information about the edges
                                    torch.Tensor(dp.authorship_tes_edge_idx),
                                    dp.tes_sections_inputs, # section batch
                                    dp.tes_sections_inputs_pos, # section pos batch
                                    torch.Tensor.float(dp.section_features),
                                    torch.Tensor(dp.section_tes_edge_idx),
                                    dp.tes_neighborhoods_representations,
                                    )

        if args.input_output_format == 1 or args.input_output_format == 42 or args.input_output_format == 43:
            pred_softmax = prediction.softmax(dim=-1) # using only softmax
        elif args.input_output_format == 2 or args.input_output_format == 3:
            pred_sigmoid = prediction.sigmoid()

    if args.input_output_format == 1 or args.input_output_format == 42 or args.input_output_format == 43:

        correct_counter = 0
        prediction_results = []
        gold = []

        # print(len(y_true), pred_softmax.shape)
        for i in range(len(y_true)):
            curr_pred = pred_softmax[i].detach()
            curr_pred = curr_pred.cpu()
            curr_pred = curr_pred.numpy() # this one is still in the form of list of prediction scores
            gold_pred = int(y_true[i])
            gold.append(gold_pred)

            top_1_pred = int(np.argmax(curr_pred))
            prediction_results.append(top_1_pred)
            if top_1_pred == gold_pred:
                correct_counter += 1

        precision = round(precision_score(gold, prediction_results, average = 'macro', zero_division = 0), 4)
        recall = round(recall_score(gold, prediction_results, average = 'macro', zero_division = 0), 4)
        acc = round(accuracy_score(gold, prediction_results), 4)

        print('Precision: {} || Recall: {} || Accuracy: {}'.format(precision, recall, acc))
        print('{} correct predictions out of {}\n'.format(correct_counter, len(dp.tes_sections_authors)))

        # saving the needed files
        # 1) prediction results in the form of its clusters
        filename = open('{}/{}_prediction_results_clusters.csv'.format(run_results_dir, test_year), 'w')
        with filename as outfile:
            write = csv.writer(outfile)
            write.writerow(prediction_results)
        all_predicted_clusters[test_year] = prediction_results # for retriever

        # 2) y_true/gold results in the form multilabel labels
        filename = open('{}/{}_y_true_multilabel.csv'.format(run_results_dir, test_year), 'w')
        with filename as outfile: # saving gold decoded
            write = csv.writer(outfile)
            write.writerows(y_true_multilabel)
        all_y_true_multilabel[test_year] = y_true_multilabel # for retriever

        filename = open('{}/{}_tra_y_true_multilabel.csv'.format(run_results_dir, test_year), 'w')
        with filename as outfile: # saving gold decoded
            write = csv.writer(outfile)
            write.writerows(y_true_multilabel_tra)

        # 3) gold results in the form of clusters
        filename = open('{}/{}_y_true_clusters.csv'.format(run_results_dir, test_year), 'w')
        with filename as outfile: # saving gold decoded
            write = csv.writer(outfile)
            write.writerow(y_true) # test y_true

        filename = open('{}/{}_tra_y_true_clusters.csv'.format(run_results_dir, test_year), 'w')
        with filename as outfile: # saving gold decoded
            write = csv.writer(outfile)
            write.writerow(y_true_tra) # train y_true

        # 4) dict of {cluster_num : [list of papers in that cluster]}
        filename = open('{}/{}_cluster_paper_list.json'.format(run_results_dir, test_year), 'w')
        json.dump(dp.CA.papers_wrt_cluster, filename)
        all_papers_wrt_cluster[test_year] = dp.CA.papers_wrt_cluster

        # 5) saving the inputs and output for current test year
        filename = open('{}/{}_inputs_papers.csv'.format(run_results_dir, test_year), 'w') # these are test papers
        with filename as outfile:
            write = csv.writer(outfile)
            write.writerows(dp.tes_sections_inputs_worded)
        all_inputs[test_year] = dp.tes_sections_inputs_worded

        # saving all train data
        filename = open('{}/{}_tra_inputs_papers.csv'.format(run_results_dir, test_year), 'w')
        with filename as outfile:
            write = csv.writer(outfile)
            write.writerows(dp.tra_sections_inputs_worded)

        filename = open('{}/{}_inputs_authors.csv'.format(run_results_dir, test_year), 'w')
        with filename as outfile:
            write = csv.writer(outfile)
            write.writerows(dp.tes_sections_authors_worded)

        filename = open('{}/{}_outputs.csv'.format(run_results_dir, test_year), 'w')
        with filename as outfile:
            write = csv.writer(outfile)
            write.writerows(dp.tes_sections_outputs_worded)
