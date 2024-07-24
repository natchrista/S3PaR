from argparse import ArgumentParser

def make_args():
    parser = ArgumentParser()

    # DATASET settings
    parser.add_argument('--input_output_format', dest = 'input_output_format', default = 1, type = int, help = '1: the original multiclass input output format, 2: multilabel input output format, 3: multilabel input output format with intentionally removed information, 42 and 43: multiple potential labels but formatting it as multiclass classification input output format, settings is the same as format 2 and 3, 22 and 32: single potential label but there exist multiple gold labels')
    parser.add_argument('--last_n_papers', dest = 'last_n_papers', default = 3, type = int, help = 'Used in the retrieval module, this is basically using only the n most recent papers from the input sequence to retrieve the recommendation')
    parser.add_argument('--embedding_mode', dest = 'embedding_mode', default = 'llm_small', type = str, help = 'Using which embedding mode, pretrained GloVe (glove) or pretrained BERT Sentence (bert) or the version without keywords embedding (incomplete_glove)')
    parser.add_argument('--use_abstract_embed', dest = 'use_abstract_embed', action = 'store_true', help = 'Using abstract emnbedding for paper node initialization')
    parser.add_argument('--use_author_embed', dest = 'use_author_embed', action = 'store_true', help = 'Using author embedding for author node initialization')
    parser.add_argument('--random_init_authors', dest = 'use_author_embed', action = 'store_false', help = 'Using random node initialization to represent author nodes')
    parser.add_argument('--use_random_init', dest = 'use_abstract_embed', action = 'store_false', help = 'Using random node initialization to represent paper nodes')
    parser.add_argument('--remove_sparse', dest = 'remove_sparse', action = 'store_true', help = 'Using data that has been filtered out from very sparse nodes and connections (based on authorship and citataion connection)')
    parser.add_argument('--keep_sparse', dest = 'remove_sparse', action = 'store_false')
    parser.add_argument('--node_init', dest = 'node_init', default = 'random', type = str, help = 'Determine the node initialization mode, pick either "random" or ...') # to do, add more node init mode, i.e., impact factor etc
    parser.add_argument('--rand_each_fold', dest = 'rand_each_fold', action = 'store_false', help = 'Wheter we do random node init every test_year')
    parser.add_argument('--use_last_rand', dest = 'use_last_rand', action = 'store_false', help = 'Whether we want to use the last randomization or create new one')
    parser.add_argument('--feat_graph_type', dest = 'feat_graph_type', default = 'separated', type = str, help = 'Determine what kind of graph the preprocessor should create. Pick either "separated" for separated feature graphs or "kg" for knowledge graph, a.k.a. large heterogeneous graph from combination of smaller feature graphs')
    parser.add_argument('--embedding_dim', dest = 'embedding_dim', default = 50, type = int, help = 'Dimension of the node embedddings')
    parser.add_argument('--anchor_nodes_num', dest = 'anchor_nodes_num', default = 25, type = int, help = 'Number of anchor nodes, this one also determines the dimension of position embedding resulting from GraphReach')
    parser.add_argument('--get_onto_hops', dest = 'onto_hops', action = 'store_true', help = 'This is for adding the information paper topics hops to "computer science" in the CSO. After hops are obtained, these hops info are then added to the text embedding of each paper. By default, the onto hops are not used, so we need to activate it using this argument')
    parser.add_argument('--get_section_embedding', dest = 'section_embedding', action = 'store_true', help = 'get positional embedding for the paper section. For each paper, there are several sections and this section embedding acts as an additional feature for the classification model to understand which section the current author is currently writing')
    parser.add_argument('--term_score_mode', dest = 'term_score_mode', default = 0, type = int, help = 'the term scoring mode for when getting onto hops information, if set to 0 then the formula is freq_ratio / 2^hops, if set to 1 then the formula is freq_ratio / hops, if set to 2 then the formula is freq_ratio * total_distance_sum, if set to 3 then the formula is freq_ratio * average_distance, if set to 4 then the formula is freq_ratio * max_distance')
    parser.add_argument('--get_coordinate_distance', dest = 'coordinate_distance', action = 'store_true', help = 'This is for adding the information of coordinate distances between one topic to another topic (i.e., current topic to its parent topic). After coordinate distances are obtained, this information is then added to the text embedding of eaach paper. By default, the coordinate distance information is not used, so we need to activate it using this argument')
    parser.add_argument('--year_range', dest = 'year_range', default = 5, type = int, help = 'the train year range given a test year, for example if a test year is 2012 and the range is 4, then the train data would contain papers from years 2011, 2010, 2009, and 2008 (up to 4 years before the test year). If year range is 0, then this means there is no year range')
    parser.add_argument('--citation_activities', dest = 'citation_activities', default = 'global', type = str, help = 'This is for data preprocessing esp during the separation of input sequence and expected multilabel outputs (remainder next activities). Given a full paper with multiple sections, each section represent one sequence. Given multiple sequences in a paper, a "global" citation activities means that the expected multilabel outputs of a current sequence considers also the remainder (next) citation activities of other sequences in the same paper, while "local" citation activities meanas that the expected multilabel outputs of a current sequence only consider the remainder (next) citation activities within the same sequence, i.e., do not use other sequences citation activities')
    parser.add_argument('--onto_important_terms_only', dest = 'onto_important_terms_only', action = 'store_true', help = 'activate to filter out ontology from less important terms. We decide a term is important when it appears many times in the abstract. The number of times is determined by a formula in main.py')
    parser.add_argument('--use_more_recent_papers_for_test', dest = 'recent_papers_for_test', action = 'store_true', help = 'minimize the ratio of older papers compared to newer papers. For example, if test year is 2018, then more papers from 2018 should exist in the data during testing step -- modification of cluster contents.')
    parser.add_argument('--older_paper_ratio', dest = 'older_paper_ratio', default = 50, type = int, help = 'ratio for older papers during test data preprocessing. The value should be between 0 to 100. By default it is set to 50, this means decrease the ratio of older paper by half.')
    parser.add_argument('--simplified_ontology_extension', dest = 'simple_onto_extend', action = 'store_true', help = 'Extend the existing ontology using a simple ontology extension method.')
    parser.add_argument('--update_paper_keywords_cache', dest = 'update_paper_keywords_cache', action = 'store_true', help = 'updates the extracted keywords cache because new keywords extraction tool is used or getting better extraction results')
    parser.add_argument('--use_neighbors_of_keywords_in_onto', dest = 'kw_neighbors_in_onto', action = 'store_true', help = 'Similar to SAGE, use the neighboring keywrods information depending on the number of hops set. This is to add additional information to the paper keywords.')
    parser.add_argument('--kw_neighbor_distance', dest = 'kw_neighbor_distance', type = int, default = 5, help = 'this is to set the max distance of neighbor information to get for each keyword in the ontology')
    parser.add_argument('--randomize_section_order', dest = 'randomize_section_order', action = 'store_true', help = 'Randomize the input instances based on its section. For example, switching section 2 with section 3, section 3 with section 1, etc')


    # GENERAL CLUSTERING settings
    parser.add_argument('--clustering_method', dest = 'clustering_method', default = 'k_means', type = str, help = 'Clustering approach used to cluster the unique papers for the current test year. the approaches can be either k_means/mini_batch_k_means/affinity_propagation/agglomerative_clustering/birch/dbscan/mean_shift/optics/spectral_clustering/spectral_clustering_cit_graph/gaussian_mixture/ontology')
    parser.add_argument('--elbow_method', dest = 'elbow_method', action = 'store_true', help = 'Finding the most optimal number of clusters using ELBOW METHOD') # only activate if I want to check the optimal number of clusters per test year
    parser.add_argument('--gap_statistics', dest = 'gap_statistics', action = 'store_true', help = 'Finding the most optimal number of clusters using GAP STATISTICS') # only activate if I want to check the optimal number of clusters per test year
    parser.add_argument('--silhouette', dest = 'silhouette', action = 'store_true', help = 'Finding the most optimal number of clusters using SILHOUETTE SCORE')
    parser.add_argument('--num_clusters', dest = 'num_clusters', default = 6, type = int, help = 'Number of clusters for the clusterer. Optimal number of clusters on the time of adding this argument is 6 (can be achieved by doing one or all of the analysis methods above, i.e., elbow_method, gap_statistics, silhouette)')

    # ONTOLOGY BASED CLUSTERING and VEC2GC settings
    parser.add_argument('--use_onto_for_vec2gc', dest = 'onto_for_vec2gc', action = 'store_true', help = 'To ACTIVATE the involvement of ontology representaiton for Vec2GC graph construction, i.e., using ontology representation involvement for graph construction during Vec2GC')
    parser.add_argument('--no_onto_for_vec2gc', dest = 'onto_for_vec2gc', action = 'store_false', help = 'To DEACTIVATE the involvement of ontology representaiton for Vec2GC graph construction, i.e., not using ontology representation whatsoever for graph construction during Vec2GC')
    parser.add_argument('--onto_graph_root', dest = 'onto_graph_root', default = 'computer science', type = str, help = 'The root of the computer science ontology, by default it is "computer science"')
    parser.add_argument('--onto_parents_level',dest = 'onto_parents_level', default = 1, type = int, help = 'SHOULD BE DECLARED WHEN USING "ONTOLOGY" CLUSTERING METHOD AND/OR "ONTO_HOPS" IS SET TO TRUE. This argument is for Which level should the number of clusters are based on. For example, in the first level of the ontology, there are a total of 30 nodes, this means the number of clusters will be 30')
    parser.add_argument('--use_skipgram', dest = 'skipgram', action = 'store_true', help = 'Using skipgram to as an additional step to the windowing mode during the finding of ontology terms from paper abstracts')
    parser.add_argument('--no_skipgram', dest = 'skipgram', action = 'store_false', help = 'Using skipgram to as an additional step to the windowing mode during the finding of ontology terms from paper abstracts')
    parser.add_argument('--skip_distance_range', dest = 'skip_distance_range', default = 2, type = int, help = 'The skip distance range for the skipgram, by default it is set to 2. This means k = [1, 2]')
    parser.add_argument('--community_detector', dest = 'community_detector', default = 'louvain', type = str, help = 'The community detector algorithm. Can choose between louvain OR greedy_modularity')
    parser.add_argument('--evaluate_community', dest = 'evaluate_community', action = 'store_true', help = 'If activated, then the community of the current test data will be evaluated. Evaluation can only be done per one test year --> cannot iterate the test years')
    parser.add_argument('--make_parentless_to_another_cluster', dest = 'parentless_to_another_cluster', action = 'store_true', help = 'If activated, another node is made to accomodate the parentless paper, i.e., there will be an aditional community specifically for these papers. Drawback is that the members of this community may most likely be random')
    parser.add_argument('--dimension_reductor', dest = 'dimension_reductor', default = 'pca', type = str, help = 'This is for the dimentionality reduction method used for the visualization and for obtaining the coordinate feature obtain class (UpdateWithCoordinatesDistanceInfo in utils.py). There are only two methods used so far, so we can use either "pca" or "tsne". By default the method used is pca')
    parser.add_argument('--coordinate_dimension', dest = 'coordinate_dimension', default = 2, type = int, help = 'The dimension of the coordinates for obtaining coordinate distance feature or for visualization. The dimension should either be 2 or 3 for now (may increase later probably)')
    parser.add_argument('--distance_formula', dest = 'distance_formula', default = 'inner_product', type = str, help = 'The formula used to calculate distance between two vectors (or coordinates), by default it is set to "inner_product" but we can choose between "euclidean" OR "inner_product"')

    # DYNAMIC ONTOLOGY SETTINGS
    parser.add_argument('--use_dynamic_onto', dest = 'dynamic_onto', action = 'store_true', help = 'Use to activate the use of dynamic ontology approach during the ontology-based clustering method')
    parser.add_argument('--ontox_classifier', dest = 'ontox_classifier', default = 'Gaussian_Naive Bayes', type = str, help = 'This is the classification method used for filtering out edges during ontology extension. Choose between Logistic_Regression/Gaussian_Naive Bayes/KNN/Decision_Tree/Random_Forest/MLP_10_Hidden/MLP_50_Hidden/MLP_100_Hidden/MLP_200_Hidden/SVM')
    parser.add_argument('--ontox_status', dest = 'ontox_status', default = 'limit1_30', type = str, help = 'Set the status for ontology extension, there are 3 choices nolimit/limit1/limit1_30')
    parser.add_argument('--dynamic_onto_year', dest = 'dynamic_onto_year', default = 2018, type = int, help = 'Deciding up to which year should the keywords inside the extended CSO exist. Dynamic_onto_year is set to the same year as the test year')
    parser.add_argument('--dynamic_onto_extend', dest = 'dynamic_onto_extend', action = 'store_true', help = 'This is to enable dynamically selecting the papers used to obtain REBEL triples which are later used to extend the ontology. Dynamism is based on, as usual, test years.')
    parser.add_argument('--term_importance', dest = 'term_importance', default = 'tfidf', type = str, help = 'This is to calculate the importance score of each ontology term. used when we want to keep important ontology terms only. Can use either tfidf or bm25')
    parser.add_argument('--bm25_representative_measure', dest = 'bm25_representative_measure', default = 'average', type = str, help = 'This is the measurement used to create the BM25 representative score. Choose either average or sum only.')
    parser.add_argument('--document_lemmatization', dest = 'document_lemmatization', action = 'store_true', help = 'use this to lemmatize the documents (abstracts and keywords) used for deciding important ontology terms')
    parser.add_argument('--use_vanilla_cso', dest = 'vanilla_cso', action = 'store_true', help = 'This is for removing all extension to the ontology and keep only topics from the original CSO')

    # CLUSTER QUALITY CHECKER settings
    parser.add_argument('--cluster_checking_mode', dest = 'cluster_checking_mode', default = 'samples_sim', type = str, help = 'samples_sim = Checking the quality of clusters resulting from the clustering method using the following method: for each cluster, get A SAMPLE OF PAPERS, and check the similarity of this sample of papers || samples_tsne = Checking the quality of clusters resulting from the clustering method using the following method: for each cluster, get A SAMPLE OF PAPERS, and check the T-SNE of this sample of papers || complete_sim : Checking the quality of clusters resulting from the clustering method using the following method: for each cluster, get ALL OF THE PAPERS inside that cluster, and check the similarity of these papers || complete_tsne * Checking the quality of clusters resulting from the clustering method using the following method: for each cluster, get ALL OF THE PAPERS inside that cluster, and check the T-SNE of these papers')
    parser.add_argument('--cc_sample_num', dest = 'cc_sample_num', default = 50, type = int, help = 'The number of papers to be sampled if cluster checking mode is one of the "samples_{}" methods')
    parser.add_argument('--cc_sim_threshold', dest = 'cc_sim_threshold', default = 0.50, type = float, help = 'The threshold to determine whether a paper is considered to be similar, default is set to 0.5 (the common threshold used for determining similarity is usually 0.5)')

    # RETRIEVER settings
    parser.add_argument('--sim_threshold', dest = 'sim_threshold', default = 0.75, type = float, help = 'Similarity threshold to find relevant papers. Papers that are less than threshold are not considered to be relevant')
    parser.add_argument('--ltr_mode', dest = 'ltr_mode', default = 'ranknet', type =  str, help = 'The learning to rank model to use. Choose either ranknet/lambdarank/lambdamart')
    parser.add_argument('--use_ltr_in_sim', dest = 'ltr_in_sim', action = 'store_true', help = 'For the similarity-based retrieval module, we can add additional learning to rank step. So the overall steps would be as follows: clustering --> classification to next papers cluster --> reranking the list of papers in the current cluster based on their relevancy (cosine similarity) --> improving the ranks of the already reranked list of papers using learning to rank')
    parser.add_argument('--lambdamart_trees', dest = 'lambdamart_trees', default = 10, type = int, help = 'The number of trees for LambdaMART LtR model')
    parser.add_argument('--use_relevance_importance', dest = 'relevance_importance', action = 'store_true', help = 'This is used during retrieval step, i.e., right after calculating a recommended paper similarity to a ground truth paper. relevance importance adds additional importance score to the similarity score, i.e., when a recommended paper is similar to early-appearing ground truth papers then this mean the importance score is higher. This is using the logic that when a paper is more similar to the early-appearing ground truth papers, this means that the paper is more closer to the supposed next paper')
    parser.add_argument('--show_recommendation_sample', dest = 'recommendation_sample', action = 'store_true', help = 'Showing recommendation results during retrieval step (retrieval module)')
    parser.add_argument('--test_data', dest = 'test_data_for_sample_show', type = str, default = None, help = 'this is to make sure the test data used is the same as the proposed method. This argument is for baselines in baselines.py')
    parser.add_argument('--evaluate_diversity_and_focus', dest = 'diversity_focus', action = 'store_true', help = 'For activating the diversity vs. focus evaluation for the retrieval results, will output some numbers showing diversity and focus measure along with heatmaps to plot these numbers')
    parser.add_argument('--max_diversity_scoring_count', dest = 'max_diversity_scoring_count', default = 200, type = float, help = 'This is for limiting the diversity scoring to only the first X results (X is set to 200 by default). This is because diversity scoring takes a very long time to do even with caching')
    parser.add_argument('--get_output_keywords', dest = 'output_keywords', action = 'store_true', help ='Getting the keywords of each recommended paper (the most similar paper in the code)')
    parser.add_argument('--run_version', dest = 'run_version', type = str, help = 'This is for making the title of the output keywords saving file')
    parser.add_argument('--partial_match_keywords', dest = 'partial_match', action = 'store_true', help = 'Partial matching the keywords not inside the ontology to the keywords existing inside the ontology')
    parser.add_argument('--onto_based_retriever_alpha', dest = 'onto_based_retriever_alpha', default = 0.5, type = float, help = 'This is the alpha value for the weighted sum formula used in the ontology based retrieval. Default is set to 0.5')

    # NN settings
    parser.add_argument('--lite_run', dest = 'lite_run', action = 'store_true', help = 'Using the lite version of S3PaR')
    parser.add_argument('--lite_gru', dest = 'lite_gru', action = 'store_true', help = 'Using lite GRU instead of the original GRU to obtain the sequence embedding, lite S3PaR has to be used for this to work')
    parser.add_argument('--use_transformer', dest = 'use_transformer', action = 'store_true', help = 'Using Transformers to obtain sequence embedding')
    parser.add_argument('--use_hftransformer', dest = 'use_hftransformer', action = 'store_true', help = 'Using Huggingface transformers library feature extractor to obtain sequence embedding')
    parser.add_argument('--hf_model', dest = 'hf_model', type = str, default = 'gpt2', help = 'For selecting which huggingface model to use, by the fault we use gpt2. Can choose other options, namely, ...')
    parser.add_argument('--use_svae', dest = 'use_svae', action = 'store_true', help = 'Use sequential VAE as explained and written in https://github.com/noveens/svae_cf/blob/master/main_svae.ipynb (paper is mentioned in there)')
    parser.add_argument('--hf_dimentionality_reductor', dest = 'hf_dimentionality_reductor', type = str, default = 'tsne', help = 'The dimentionality reductor method chosen to reduce the dimension of resulting Huggingface-related embeddings')
    parser.add_argument('--in_size', dest = 'in_size', default = 300, type = int, help = 'node feature dimension')
    parser.add_argument('--hidden_size1', dest = 'hidden_size1', default = 300, type = int, help = 'first hidden layer dimension')
    parser.add_argument('--hidden_size2', dest = 'hidden_size2', default = 300, type = int, help = 'second hidden layer dimension')
    parser.add_argument('--nn_model_out_size', dest = 'nn_model_out_size', default = 300, type = int, help = 'the output dimension from the nn model, this is different from RecModel\'s output dimension, which is based on the number of classes')
    parser.add_argument('--train_onto_sage', dest = 'train_onto_sage', action = 'store_true', help = 'Training graph SAGE for ontology. this model will be used for the retrieval step later')

    # RETRIEVER MODULE settings
    parser.add_argument('--lite_retriever', dest = 'lite_retriever', action = 'store_true', help = 'Using lite version of retriever module')
    parser.add_argument('--ontology_based_retriever', dest = 'ontology_based_retriever', action = 'store_true', help = 'Using ontology based version of retriever module, here ontology information is considered for retrieval. Lite retriever version is used here')

    # TRAINING settings
    parser.add_argument('--test_per_epoch', dest = 'test_per_epoch', action = 'store_true', help = 'Do evaluation per EPOCH using test data')
    parser.add_argument('--test_per_batch', dest = 'test_per_batch', action = 'store_true', help = 'Do evaluation per BATCH using test data')
    parser.add_argument('--auto_detect_device', dest = 'auto_detect_device', action = 'store_true', help = 'If set to true then the code will automatically detect whether there is GPU or not. If there is GPU then the code will automatically be set to run with CUDA, otherwise CPU. When this argument is set to False, then the code will be run on CPU')

    # ABLATION test settings
    parser.add_argument('--use_similarity', dest = 'use_similarity', action = 'store_true', help = 'Use similarity measures for the text embeddings')
    parser.add_argument('--ablation_no_sage', dest = 'ablation_no_sage', action = 'store_true', help = 'For performing ablation test, removing SAGE layer from the model')
    parser.add_argument('--use_gru_mha', dest = 'use_gru_mha', action = 'store_true', help = 'Use this to activate the use of MHA in GRU')
    parser.add_argument('--no_gru_mha', dest = 'use_gru_mha', action = 'store_false', help = 'Use this to deactivate the use of MHA in GRU')
    parser.add_argument('--use_external_attention', dest = 'use_external_attention', action = 'store_true', help = 'Use this to activate the use of External Attention')
    parser.add_argument('--use_n2v', dest = 'use_n2v', action = 'store_true', help = 'Use this to activate the use of Node2Vec as a replacement for Graph SAGE. Note that SAGE has to be deactivated to use this')
    parser.add_argument('--use_position_encoding', dest = 'use_position_encoding', action = 'store_true', help = 'Use positional encoding for the MHA part')
    parser.add_argument('--no_position_encoding', dest = 'use_position_encoding', action = 'store_false', help = 'Remove the use of positional encoding for the MHA part')
    parser.add_argument('--use_mha', dest = 'use_mha', action = 'store_true', help = 'Use MHA for the prediction layer')
    parser.add_argument('--no_mha', dest = 'use_mha', action = 'store_false', help = 'Disable MHA for the prediction layer')
    parser.add_argument('--use_gru', dest = 'use_gru', action = 'store_true', help = 'Use this to activate the use of GRU')
    parser.add_argument('--no_gru', dest = 'use_gru', action = 'store_false', help = 'Use this to completely deactivate the use of GRU')
    parser.add_argument('--use_contexts', dest = 'use_contexts', action = 'store_true', help = 'Use episodic context and category context just like what is done in CoCoRec paper')
    parser.add_argument('--use_cf', dest = 'use_cf', action = 'store_true', help = 'Use Collaborative Filtering. If set True then use the same method (Collaboration Module) explained in CoCoRec paper')
    parser.add_argument('--use_cluster_emb', dest='use_cluster_emb', action = 'store_true', help = 'Use cluster embedding to create neighborhood representation for the collaborative filtering approach. If not activated, then abstract embeddings are used to represent the papers in each potential neighbor\'s sequence')
    parser.add_argument('--use_cosine_sim_for_edges', dest = 'cosine_sim_for_edges', action = 'store_true', help = 'This allows the use of similarity for the in-cluster edge finding in the ontology-based cluser. this is the 1.2 step ')
    parser.add_argument('--no_cosine_sim_for_edges', dest = 'cosine_sim_for_edges', action = 'store_true', help = 'This deactivate the use of similarity for the in-cluster edge finding in the ontology-based cluser. this is the 1.2 step ')
    parser.add_argument('--triple_types_experiment', dest = 'triple_types_experiment', action = 'store_true', help = 'This is to conduct separate experiment for type 1 REBEL triples or type 2 REBEL triples')

    # DEEP NEURAL NETWORK TRAINING settings
    parser.add_argument('--train_test', dest = 'train_status', action = 'store_true', help = 'Wheter do train and test or just test, if just test then use --test')
    parser.add_argument('--test', dest = 'train_status', action = 'store_false', help = 'Wheter do train and test or just test, if just test then use --test')
    parser.add_argument('--use_last_model', dest = 'use_last_model', action = 'store_true', help = 'Whether using last model for testing or not')
    parser.add_argument('--use_dgi', dest = 'use_dgi', action = 'store_true', help = 'Wheter use Deep Graph Infomax or not')
    parser.add_argument('--not_use_dgi', dest = 'use_dgi', action = 'store_false', help = 'Wheter use Deep Graph Infomax or not')
    parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', help = 'Whether using GPU')
    parser.add_argument('--cpu', dest = 'gpu', action = 'store_false', help = 'Whether using CPU')
    parser.add_argument('--cuda', dest = 'cuda', default = '0', type = str)
    parser.add_argument('--train_status', dest = 'train_status', action = 'store_true')
    parser.add_argument('--nn_model', dest = 'nn_model', default = 'sage', type = str, help = 'Pick either GraphSAGE (sage), GraphReach (gr) OR R-GCN (rgcn)')
    parser.add_argument('--gr_feature', dest = 'gr_feature', default = 'position', type =  str, help = 'If using GraphReach model, what kind of feature embedding to use. Pick either "position" or "structure"')
    parser.add_argument('--epoch', dest = 'epoch', default = 30, type = int, help = 'The number of epochs for training')
    parser.add_argument('--batch_size', dest = 'batch_size', default = 8, type = int)
    parser.add_argument('--learning_rate', dest = 'learning_rate', default = 0.01, type = float)

    # BASELINE SETTINGS
    parser.add_argument('--baseline_mode', dest = 'baseline_mode', default = 'cf', type = str, help = 'Define either cf/cbf/ncf/simple_sequential for vanilla CF, vanilla CBF, Neural CF, and Simple Sequential Recommendation respectively')

    # SETTINGS FOR RUNNING PICKLES AND RETRIEVAL MODULE PERMUTATION TEST PREP -- used mainly for main_pickle.py and permuation_test_retrieval_prep.py
    parser.add_argument('--run_results_subfolder', dest = 'run_results_subfolder', type = str, default = '')
    parser.add_argument('--run_results_filename', dest = 'run_results_filename', type = str, default = '')
    parser.add_argument('--starting_epoch', dest = 'starting_epoch', type = int, help = 'This tells the code from which epoch should the training start. For example, if a process got killed during the 9th epoch, then we should set the starting epoch as 9 so it can start from the 9th epoch')
    parser.add_argument('--last_run_highest_precision', dest = 'last_run_highest_precision', type = float, default = None, help = 'The highest precision resulted by the model before its training process got killed')
    parser.add_argument('--single_test_year', dest = 'single_test_year', type = int, default = 2017, help = 'This is used for the main_pickle.py, In the case of this code, we should only run using ONE test year at a time')
    parser.add_argument('--number_of_runs', dest = 'number_of_runs', type = int, default = 5, help = 'This is used for the permutation_test_retrieval_prep.py code, for test year 2009, we can only run a few runs at a time because one run requires around 3 to 4 hours')
    parser.add_argument('--number_of_past_runs', dest = 'number_of_past_runs', type = int, help = 'This is the number of runs that have been done previously, the purpose is for the code to not repeat a run, instead, continue the remaining runs')
    parser.add_argument('--pipeline_version', dest = 'pipeline_version', type = str, default = 'with_SAGE')
    parser.add_argument('--job_submission_count', dest = 'job_submission_count', type = str, default = '0', help = 'how many jobs have been submitted in the supercomputer. This is for naming purpose, the final file containing precision and ndcg scores will be titled //with_SAGE_2009_2, with_SAGE_2009_3// depending on the number of jobs already run')

    # DEFAULT SETTINGS
    parser.set_defaults(rand_each_fold = False,
                        lite_run = False,
                        lite_retriever = False,
                        ontology_based_retriever = False,
                        lite_gru = False,
                        use_transformer = False,
                        use_hftransformer = False,
                        hf_model = 'gpt2', # transformer model to be used from huggingface library
                        train_onto_sage = False,
                        use_svae = False,
                        use_gru = False,
                        use_mha = False,
                        use_gru_mha = False,
                        use_external_attention = False,
                        use_n2v = False,
                        use_position_encoding = True,
                        section_embedding = False,
                        embedding_mode = 'bert',
                        use_abstract_embed = True,
                        use_author_embed = True,
                        simple_onto_extend = False,
                        onto_hops = False,
                        onto_important_terms_only = False,
                        document_lemmatization = False,
                        term_importance = 'tfidf',
                        term_score_mode = 0,
                        coordinate_distance = False,
                        dimension_reductor = 'pca',
                        distance_formula = 'inner_product',
                        remove_sparse = True,
                        train_status = True,
                        use_last_rand = False,
                        test_per_epoch = False,
                        test_per_batch = False,
                        recent_papers_for_test = False,
                        auto_detect_device = False,
                        ablation_no_sage = False,
                        use_similarity = False,
                        gpu = False,
                        use_dgi = False,
                        use_contexts = False,
                        use_cf = False,
                        use_cluster_emb = False,
                        use_last_model = False,
                        feat_graph_type = 'separated',
                        in_size = 384, # size for bert embedding
                        hidden_size1 = 384,
                        hidden_size2 = 384,
                        nn_model_out_size = 384,
                        nn_model = 'sage',
                        epoch = 30,
                        gap_statistics = False,
                        elbow_method = False,
                        silhouette = False,
                        onto_for_vec2gc = True,
                        num_clusters = 6,
                        sim_threshold = 0.75,
                        cluster_checking_mode = 'samples_sim',
                        cc_sample_num = 50, # number of samples to take from the cluster for cluster checker
                        cc_sim_threshold = 0.5, # threshold for cluster checker
                        skipgram = True,
                        skip_distance_range = 1, # set the skip distance to 1 for now
                        evaluate_community = False,
                        cosine_sim_for_edges = True,
                        parentless_to_another_cluster = False,
                        ltr_in_sim = False, # this is to activate similarity based retrieval module + learning to rank
                        lambdamart_trees = 10,
                        relevance_importance = False,
                        recommendation_sample = False,
                        dynamic_onto = False,
                        dynamic_onto_extend = False,
                        triple_types_experiment = False,
                        vanilla_cso = False,
                        update_paper_keywords_cache = False,
                        output_keywords = False,
                        partial_match = False,
                        randomize_section_order = False,
                        )

    args = parser.parse_args()
    return args
