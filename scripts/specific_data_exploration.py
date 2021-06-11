import pandas as pd
import os
import re
from collections import Counter, defaultdict

# TODO: re-evaluate whether this is even necessary
fdf_location = 'file_level_data.tsv'  # location of the file-level data
adf_location = 'annotation_level_data.tsv'  # and of the annotation-level data

fdf = pd.read_csv(fdf_location, sep='\t')
adf = pd.read_csv(adf_location, sep='\t')

fdf_polnear = fdf[fdf.source == 'PoLNeAR']
adf_polnear = adf[adf.source == 'PoLNeAR']

fdf_parc = fdf[fdf.source == 'PARC']
adf_parc = adf[adf.source == 'PARC']


# Is advmod the most frequent dep label in the cue spans with say or ROOT?

def most_freq_item_in_chain(fdf, adf, main_col, main_val, target_col, chain,
                            no_items=5):
    '''Finds the most frequent co-occuring item of a given type with some other
       type in a chain. So for example, for all cues of length > 1, if one cue
       in this chain is a lemma ‘say’, find the most common dep labels of other
       cues in that same chain.

       :param fdf: the file-level dataframe
       :param adf: the annotation-level dataframe
       :param main_col: the column in the adf that contains the main values, so
                        lemmas in the example, columns are as follows:
                        # Actually in hindsight these arent accurate anymore, working on correcting them right now
                        0 - file name
                        1 - token
                        2 - lemma
                        3 - micro pos (i.e., VBG instead of V)
                        4 - macro pos (i.e., V instead of VBG)
                        5 - dep label
                        6 - target values (CUE, CONTENT, SOURCE, O)
                        7 - source (i.e., PARC or PoLNeAR)
                        8 - type (i.e., dev or train)
        :param main_val: the value you’re searching for, ‘say’ in the example
        :param target_col: column the question target is in, so dep labels in
                           the example
        :param chain: ‘CUE’, ‘CONTENT’, or ‘SOURCE’, depending on what type of
                      chain you are looking for, or with a B-/I- prefix
        :param no_items: the top X of most frequent items the function returns
    '''
    # First start off with figuring out which files have more, say, cues than
    # they have chains, since those files would contain multi-token cues. This
    # solution is imperfect, especially for cues, since not all chains contain
    # all three possible types.
    
    chain_name = chain.lower() if chain[1] != '-' else chain[2:].lower()
    column_name = chain_name.lower()+'_count'  # get the relevant column name
    # and extract the right rows from fdf
    multi_token_chains = fdf.loc[fdf[column_name] > fdf.chain_count]
    relevant_chains = []
    for _, row in multi_token_chains.iterrows():
        with open(row.file, encoding='utf8') as f:
            chains_in_file = defaultdict(list)
            for line in f.readlines():
                line = line.rstrip().split('\t')
                if len(line) > 1 and chain in line[-1]:
                    ids = {i for i in re.findall('[0-9][0-9]*', line[-1])}
                    for i in ids:
                        chains_in_file[i].append((line[main_col], line[target_col]))  # noqa
            for i in chains_in_file:
                if len(chains_in_file[i]) > 1:
                    for j in chains_in_file[i]:
                        if j[0] == main_val:
                            other_vals = [jj[1] for jj in chains_in_file[i] if jj[0] != main_val]  # noqa
                            relevant_chains += other_vals
                            break
    counts = Counter(relevant_chains)
    counts = [(i[0], i[1] / len(relevant_chains)) for i in counts.most_common(no_items)]  # noqa
    for count in counts:
        print(count)


most_freq_item_in_chain(fdf_polnear, adf_polnear, -3, 'root', -3, 'CUE')
