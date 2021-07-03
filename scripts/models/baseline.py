import re
import pandas as pd
import os
import networkx as nx

def generate_filepaths(path_to_directory, corpus, dataset):
    '''
    Generate a list with the filepaths to each article
    :return: list of filepath strings
    '''

    file_list = []
    for subdir, dirs, files in os.walk(path_to_directory, corpus):
        # DRI: second condition is for some reason necessary for the script to run in my machine
        if len(files) > 0 \
        and files[0] != '.DS_Store' \
        and dataset in subdir \
        and corpus in subdir:
            file_list += [os.path.join(subdir, file) for file in files]

    return file_list

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
        agg_func = lambda s: [(ar, s_n, i, t, l, p, d, h, a) for ar, s_n, i, t, l, p, d, h, a in zip(
                                                                     s["article"].values.tolist(),
                                                                     s["sent_n"].values.tolist(),
                                                                     s["sent_idx"].values.tolist(),
                                                                     s["token"].values.tolist(),
                                                                     s["lemma"].values.tolist(),
                                                                     s["pos"].values.tolist(),
                                                                     s["dep_label"].values.tolist(),
                                                                     s["dep_head"].values.tolist(),
                                                                     s["att"].values.tolist(),
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
        head_index = word[-2]
        dep_index = word[2]
        # dep_label = word[-4]
        # if dep_label != 'ROOT':
        #     edges.append((head_index, dep_index))
        edges.append((head_index, dep_index))
    graph = nx.DiGraph(edges)

    return graph

def generate_baseline(sentences, cue_gzt):
    """
    Implement syntactic, gazetteer-based baseline for cue, source and content extraction.
    :param sentences: list of sentence objects, where each token is represented as a list of corresponding cells in data frame.
    :param cue_gzt: list of cue words from literature.
    :return: a nested dictionary, in which the first-level key is the article filename, the second-level key is the sentence number, and the third-level key is the token number. The corresponding predictions are the lowest-level value.
    """

    predictions = dict() # top level dict with articles as keys

    for s in sentences:

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
                    head = t[-2]
                    dep_label = t[-3]
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
        article = s[0][0]
        sent_number = s[0][1]
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


    return predictions

def generate_attribution_column(df,pred):
    '''
    Given the input data frame and a dictionary of predictions (in which article is the first level,
    sentences the second level, tokens the third level, and tags the lowest level values), generate a list of attribution tags.
    :param df: pandas data frame resulting from reading the article file as with pd.read_csv().
    :param pred: the dictionary with article, sentences, tokens and respective tags as output by baseline system.
    :return: a list of attributation tags, where each tag corresponds to a token classification by baseline in conll format e.g. ___I-CONTENT. Each underscore represents an AR in the article.
    '''

    attributions = []  # the column with attributions in output file
    for article in df.article.unique(): # this is not necessary anymore, as each article is read in as a separate data frame
        # find how many cues are there per article to add same number of subcolumns to attribution column
        n_cues = 0
        for sent, tokens in pred[article].items():
             if "B-CUE" in tokens.values():
                n_cues += 1
        if n_cues == 0: # prevent error that rises when evaluating article in which baseline does not find any ARs
            n_cues = 1
        # add tags to the subcolumn of its AR
        df_article = df.loc[df.article == article]
        subcol_idx = 0
        AR_idx = 1 # index of AR in article to attach to token tag
        for sent in df_article.sent_n.unique():
            # tags = pred[article][sent].values()
            sorted_items = sorted(pred[article][sent].items()) # sort sentence dict keys by token index in ascending order to match their order in the data frame
            tags = [item[1] for item in sorted_items]
            if any(label in tags for label in ['B-SOURCE', 'B-CONTENT', 'B-CUE', 'I-SOURCE', 'I-CONTENT', 'I-CUE']):
                for tag in tags:
                    att = ['_' for cue in range(n_cues)]  # add as many subcolumns as the number of cues in the article
                    if tag != '_': # if token is an AR element, add the token tag to the subcolumn at position subcol_idx with AR index e.g.__B-SOURCE-3
                        att[subcol_idx] = tag + '-' + str(AR_idx)
                    else: # if token is outside AR, just add the underscore
                        att[subcol_idx] = tag
                    attributions.append(' '.join(att))  # join subcolumns into a string separated with whitespaces
                subcol_idx += 1  # increase subcolumn position for the next AR
                AR_idx += 1 # increase AR idx to attach to tags for the next AR
            else:  # if the sentence has no ARs, just add attribution with underscores
                for tag in tags:
                    att = ['_' for cue in range(n_cues)]
                    attributions.append(' '.join(att))

    return attributions


# generate full data frame (all articles from a corpus concatenated)
corpus = "polnear"
dataset = 'dev'
files = generate_filepaths(f'../../data/{corpus}-conll', corpus, dataset)
for file in files:
    # df["gold"] = df["att"].apply(extract_gold_label) # strip underscores and unwanted labels from attribution column
    column_names = ['article',
                    'sent_n',
                    'doc_idx',
                    'sent_idx',
                    'offsets',
                    'token',
                    'lemma',
                    'pos',
                    'dep_label',
                    'dep_head',
                    'att']
    df = pd.read_csv(file, sep='\t', names=column_names)
    df.dropna(how="all", inplace=True)

    # check most frequent dep_labels for each gold label to support syntactic baseline dev
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df.groupby("gold")["dep_label"].value_counts(normalize=True))

    # create sentence instances
    getter = SentenceGetter(df)
    sentences = getter.sentences

    # collect gold labels
    # gold = [[token[-1] for token in sentence] for sentence in getter.sentences]

    # read in list of reporting verbs from literature
    cue_gzt = pd.read_csv('../../data/cue_list.csv')["cue"].tolist()

    # generate syntactic baseline
    pred = generate_baseline(sentences, cue_gzt)

    # create attribution column
    attribution_column = generate_attribution_column(df,pred)
    output_file = df.copy()
    output_file["att"] = attribution_column

    # select float columns only and convert them into ints
    float_col = output_file.select_dtypes(include=['float64'])
    for col in float_col.columns.values:
        output_file[col] = output_file[col].astype('int64')

    # write out baseline output file
    df_rows = output_file.values.tolist()
    rows_out = []
    for row in df_rows:
        rows_out.append(row)
    with open(f'../../data/output/baseline/{dataset}_{corpus}/{os.path.basename(file)}.system', 'w') as tsvfile:
        for row in rows_out:
            if row[3] == 1 and row != rows_out[0]:
                tsvfile.write('\n') # add blank lines between sentences
            row = [str(cell) for cell in row]
            line = '\t'.join(row) + '\n'
            tsvfile.write(line)