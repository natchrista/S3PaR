from retriever import Retriever, RetrieverLite
from retriever_ver2 import RetrieverLiteOnto
import json, csv
from tqdm import tqdm
from args import *
from learning_to_ranks import RankNet, LambdaRank, LambdaMart
from sentence_transformers import SentenceTransformer # for SBERT

args = make_args()
print('\n', args, '\n')

similarity = True

result_subfolder = 'ep10_iomode42_b32_lr00001_20240108185611499845' # 

test_years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]

last_n_papers = 3 # 3 most recent papers from the input sequence
all_k = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
similarity_threshold = 0.75

relevance_range = 2 # binary value, if more than threshold then True (1) otherwise False (0)

paper_keywords = json.load(open('../data/paper_corresponding_topics_yake.json'))
paper_abstract = json.load(open('../data/paper_paper_abstract_title_bert.json'))
sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

all_papers_wrt_cluster = {} 

# DATA FOR TRAINING -- just in case I want to train from zero and not load an existing model
all_inputs_tra = {} 
all_y_true_clusters_tra = {} 
all_y_true_multilabel_tra = {} 

# DATA FOR TESTING
all_predicted_clusters = {} 
all_inputs = {} 
all_authors = {}
all_y_true_clusters = {} 
all_y_true_multilabel = {} 

try:
	paper_and_year = json.load(open('../data/paper_and_year.json'))
	print('Loaded paper and year information...\n')
except:
	print('Paper and year information does not exist. Creating one now...\n')
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
print()

# loading all the needed files above
for test_year in test_years:

	filename = open('run_results/{}/{}_prediction_results_clusters.csv'.format(result_subfolder, test_year))
	reader = csv.reader(filename, delimiter = ',') # all predicted clusters for test
	curr_predicted_clusters = []
	for item in reader:
		curr_predicted_clusters.append(item)
	curr_predicted_clusters = [int(item) for item in curr_predicted_clusters[0]]
	all_predicted_clusters[test_year] = curr_predicted_clusters

	# test data input
	filename = open('run_results/{}/{}_inputs_papers.csv'.format(result_subfolder, test_year))
	reader = csv.reader(filename) # inputs for test
	curr_input_papers = []
	for item in reader:
		curr_input_papers.append(item)
	all_inputs[test_year] = curr_input_papers

	filename = open('run_results/{}/{}_tra_inputs_papers.csv'.format(result_subfolder, test_year)) # inputs for train
	reader = csv.reader(filename)
	curr_input_papers = [] # input papers from the train data
	for item in reader:
		curr_input_papers.append(item)
	all_inputs_tra[test_year] = curr_input_papers

	if show_recommendation_samples:
		filename = open('run_results/{}/{}_inputs_authors.csv'.format(result_subfolder, test_year))
		reader = csv.reader(filename)
		curr_input_authors = []
		for item in reader:
			curr_input_authors.append(item)
		all_authors[test_year] = curr_input_authors
	else:
		all_authors = None

	filename = open('run_results/{}/{}_y_true_clusters.csv'.format(result_subfolder, test_year))
	reader = csv.reader(filename, delimiter = ',') # y true clusters for test
	curr_y_true_clusters = []
	for item in reader:
		curr_y_true_clusters.append(item)
	curr_y_true_clusters = [int(item) for item in curr_y_true_clusters[0]]
	all_y_true_clusters[test_year] = curr_y_true_clusters

	filename = open('run_results/{}/{}_tra_y_true_clusters.csv'.format(result_subfolder, test_year))
	reader = csv.reader(filename, delimiter = ',') # y true clusters for train
	curr_y_true_clusters = []
	for item in reader:
		curr_y_true_clusters.append(item)
	curr_y_true_clusters = [int(item) for item in curr_y_true_clusters[0]]
	all_y_true_clusters_tra[test_year] = curr_y_true_clusters

	# test data gold
	filename = open('run_results/{}/{}_y_true_multilabel.csv'.format(result_subfolder, test_year))
	reader = csv.reader(filename) # y true multilabel for test
	curr_y_true = []
	for item in reader:
		curr_y_true.append(item)
	all_y_true_multilabel[test_year] = curr_y_true

	filename = open('run_results/{}/{}_tra_y_true_multilabel.csv'.format(result_subfolder, test_year))
	reader = csv.reader(filename) # y true multilabel for train
	curr_y_true = []
	for item in reader:
		curr_y_true.append(item)
	all_y_true_multilabel_tra[test_year] = curr_y_true

	filename = open('run_results/{}/{}_cluster_paper_list.json'.format(result_subfolder, test_year))
	temp = json.load(filename)
	curr_papers_wrt_clusters = {}
	for k, v in temp.items():
		curr_papers_wrt_clusters[int(k)] = v
	all_papers_wrt_cluster[test_year] = curr_papers_wrt_clusters

ltr_model = None # without learning to rank 

retriever = Retriever(all_predicted_clusters,
					  all_inputs,
					  all_y_true_multilabel,
					  all_papers_wrt_cluster,
					  result_subfolder,
					  args,
					  last_n_papers,
					  test_years = test_years,
					  all_k = all_k,
					  threshold = similarity_threshold,
					  relevance_range = relevance_range,
					  use_learn_to_rank = args.ltr_in_sim,
					  ltr_model = ltr_model,
					  input_authors = all_authors,
					  use_similarity_based = similarity,
					  use_knn = knn,
					  use_kdtree = kdtree,
					  show_recommendation_samples = show_recommendation_samples,
					  compare_keywords_with_onto = compare_keywords_with_onto,
					  diversity_focus = diversity_focus,
					  )
