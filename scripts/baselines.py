import re
import pandas as pd
import os

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

# Separate tokens into sentences
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(t, s_n, l, p, d, h, a, g) for t, s_n, l, p, d, h, a, g in zip(s["token"].values.tolist(),
                                                                     s["sent_n"].values.tolist(),
                                                                     s["lemma"].values.tolist(),
                                                                     s["pos"].values.tolist(),
                                                                     s["dep_label"].values.tolist(),
                                                                     s["dep_head"].values.tolist(),
                                                                     s["att"].values.tolist(),
                                                                     s["gold"].values.tolist()
                                                                     )]
        self.grouped = self.data.groupby("sent_n").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def extract_gold_label(cell):
    '''
    Strip underscores and info attached to gold label (e.g. -PD-0).
    :return: gold labels
    '''
    match = re.search(r'[BI]-[A-Z]*', str(cell))
    if match:
        cell = match.group(0)
    else: # if cell only contains underscores, token does not belong to a source, a cue or a content
        cell = '_'
    return cell

def generate_baseline():
    pass


df = read_in_files('../../data_ar', 'parc')
df["gold"] = df["att"].apply(extract_gold_label) # strip underscores and unwanted labels from attribution column
df.to_csv('../data/full_train_dataset_parc.tsv',sep='\t')
# df = pd.read_csv('../data/full_train_dataset_parc.tsv',sep='\t')
# getter = SentenceGetter(df)
# sentences = getter.sentences


