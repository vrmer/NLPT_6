import re
import pandas as pd
import os
import networkx as nx

def generate_train_filepaths(path_to_directory, corpus):
    '''
    Generate a list with the filepaths to each training article
    :return: list of filepath strings
    '''

    file_list = []
    for subdir, dirs, files in os.walk(path_to_directory, corpus):
        # DRI: second condition is for some reason necessary for the script to run in my machine
        if len(files) > 0 \
        and files[0] != '.DS_Store' \
        and 'test' not in subdir \
        and 'dev' not in subdir\
        and corpus in subdir:
            file_list += [os.path.join(subdir, file) for file in files]

    return file_list

def read_in_files (path_to_directory, corpus):
    '''
    Read in files in given directory as data frames and concatenate them to form an unique data frame
    :param path_to_directory: the path to directory where the folders 'parc' and 'polnear' are located in your local machine.
    :param corpus: the name of the corpus (that should be part of the name of the folder in which the data files are stored)
    :return: the whole data as one pandas data frame
    '''

    file_list = generate_train_filepaths(path_to_directory, corpus)
    full_df = pd.DataFrame()
    for file in file_list:
        data = pd.read_csv(file, sep='\t', names=['article',
                                                    'sent_n',
                                                    'doc_idx',
                                                    'sent_idx',
                                                    'offsets',
                                                    'token',
                                                    'lemma',
                                                    'pos',
                                                    'dep_label',
                                                    'dep_head',
                                                    'att'])
        full_df = pd.concat([full_df,data])


    return full_df

def extract_gold_label(cell):
    '''
    Strip underscores and info attached to gold label (e.g. -PD-0).
    :return: gold labels
    '''
    match = re.findall(r'[BI]-[A-Z]*', str(cell))
    if match:
        for tag in match:
            if "-NE" not in tag: # exclude nested attributions
                cell = tag
    else: # if cell only contains underscores, token does not belong to a source, a cue or a content
        cell = '_'
    return cell

# Separate tokens into sentences
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(ar, s_n, i, t, l, p, d, h, a, g) for ar, s_n, i, t, l, p, d, h, a, g in zip(
                                                                     s["article"].values.tolist(),
                                                                     s["sent_n"].values.tolist(),
                                                                     s["sent_idx"].values.tolist(),
                                                                     s["token"].values.tolist(),
                                                                     s["lemma"].values.tolist(),
                                                                     s["pos"].values.tolist(),
                                                                     s["dep_label"].values.tolist(),
                                                                     s["dep_head"].values.tolist(),
                                                                     s["att"].values.tolist(),
                                                                     s["gold"].values.tolist()
                                                                     )]
        self.grouped = self.data.groupby(["article","sent_n"]).apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def get_graph (sentence):

    """
    Given a sentence, create networkx graph with edges corresponding to head-dependent links.
    :param sentence: sentence object.
    :return: networkx graph.
    """

    edges = []  # a  list of tuples with dep links, represented as head and dependent and their respective positions
    for word in sentence:
        head_index = word[-3]
        dep_index = word[2]
        # dep_label = word[-4]
        # if dep_label != 'ROOT':
        #     edges.append((head_index, dep_index))
        edges.append((head_index, dep_index))
    graph = nx.DiGraph(edges)

    return graph

def generate_baseline(sentences, cue_gzt):

    predictions = dict() # top level dict with articles as keys

    for s in sentences[0:5]:

        pred_dict = dict() # lower level dict with token index as keys

        # try to find cues by comparing each token lemma with the cue gazetteer
        for token in s:

            lemma = token[3]
            idx = token[2]

            if lemma in cue_gzt and "CUE" not in pred_dict.values():

                pred_dict[idx] = 'CUE'  # if they match, tag token as cue
                # create networkx graph to more easily access dep structure
                graph = get_graph(s)

                # loop through tokens again to find dependents
                for t in s:

                    # if token is dependent and dep label is advmod, neg, aux or mwe, then it is part of the CUE span
                    head = t[-3]
                    dep_label = t[-4]
                    t_idx = t[2]
                    if head == idx and dep_label in ["advmod","neg","aux","mwe"]:
                        pred_dict[t_idx] = 'CUE'

                    # if token is dependent and dep label is its subject, then it is its SOURCE.
                    elif head == idx and dep_label == "nsubj":
                        pred_dict[t_idx] = 'SOURCE'
                        # its direct and indirect dependents compose the source span.
                        dependents = nx.descendants(graph,t_idx)
                        for dep in dependents:
                            pred_dict[dep] = 'SOURCE'

                    # if token is dependent and dep label is its clausal complement, then it is its CONTENT.
                    elif head == idx and dep_label == "ccomp":
                        pred_dict[t_idx] = 'CONTENT'
                        # its direct and indirect dependents compose the content span
                        dependents = nx.descendants(graph, t_idx)
                        for dep in dependents:
                            pred_dict[dep] = 'CONTENT'

            else:
                if idx not in pred_dict.keys():
                    pred_dict[idx] = '_'

        # Add BIO-tags
        tags = list(pred_dict.values())
        counter = 0
        for tag in tags:
            if tag != '_':
                if tag not in tags[:counter]:
                    pred_dict[counter+1] = f'B-{tag}'
                else:
                    pred_dict[counter+1] = f'I-{tag}'
            counter += 1

        # add token tags to dictionary of corresponding article and sentence
        article = s[0][0]
        sent_number = s[0][1]
        if article not in predictions.keys():
            pred_sent = {sent_number : pred_dict} # second level dict with sentence index as keys
            predictions[article] = pred_sent
        else:
            predictions[article][sent_number] = pred_dict

    print(predictions)

    return predictions


# generate full training data frame
corpus = "polnear"
# df = read_in_files('../../data_ar', corpus)
# df["gold"] = df["att"].apply(extract_gold_label) # strip underscores and unwanted labels from attribution column
# df.to_csv(f'../data/full_train_dataset_{corpus}.tsv',sep='\t')

# read in full training data frame
df = pd.read_csv(f'../data/full_train_dataset_{corpus}.tsv',sep='\t')

# check most frequent dep_labels for each gold label to support syntactic baseline dev
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df.groupby("gold")["dep_label"].value_counts(normalize=True))

# create sentence instances
getter = SentenceGetter(df)
sentences = getter.sentences

# collect gold labels
gold = [[token[-1] for token in sentence] for sentence in getter.sentences]

# read in list of reporting verbs from literature
cue_gzt = pd.read_csv('../data/cue_list.csv')["cue"].tolist()

# generate syntactic baseline
pred = generate_baseline(sentences, cue_gzt)

# generate baseline output file in conll format
# find how many cues are there per article to add same number of subcolumns to attribution column
attributions = []
for article in df.article.unique():
    if article in pred.keys():
        n_cues = 0
        print(article)
        print(pred[article])
        for sent, tokens in pred[article].items():
            for tag in tokens.values():
                if "CUE" in tag:
                    n_cues += 1
        # add as many underscores as the number of cues in the article
        att = ''
        for cue in range(n_cues):
                att = att+'_ '
        # add tags to the subcolumn of its AR
        for sent, tokens in pred[article].items():
            for tag in tokens.values():
                if tag != '_':
                    pass
            for token in tokens:
                attributions.append(att)
                print(att)


