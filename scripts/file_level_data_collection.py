import os
import pandas as pd
import re
# # from collections import Counter
# # from itertools import product

# On my PC, this script walks through a directory containing only the parc and
# polnear folders. I did not double-check whether such a folder exists in this
# repository, but without such a directory this script will not work properly.

# All (two) references to directories can be found by
# searching for TODOs.

# Start out with collecting all files into a single column
file_list = []

# TODO: make sure this points to a directory containing only the parc and/or
#       polnear folders
# for subdir, dirs, files in os.walk('..\\data\\'):
for subdir, dirs, files in os.walk('../../data_ar'): # DRI: this is the filepath to the folder with the data in my local machine
    if len(files) > 0 and files[0] != '.DS_Store': # DRI: second condition is for some reason necessary for the script to run in my machine
        file_list += [os.path.join(subdir, file) for file in files]
        print(subdir)
        print(dirs)
        print(files)

df = pd.DataFrame({'file': file_list}, columns=['file'])
# Extract further information from the file path
df['source'] = df.file.apply(lambda x: 'PARC' if 'parc' in x else 'PoLNeAR')
df['type'] = df.file.apply(lambda x: re.findall('(test|train|dev)', x)[0])


def collect_sent_info(file, sep='\t'):
    '''
    Takes a file and collects all sentence- (and chain-)related information.

    :return sent_count: the number of sentences
    :return tok_count: the total number of tokens
    :return all_chains: a set containing all chains in the file, the function
                        returns its length as it corresponds to the number of
                        chains in the file
    :return chains_spanning_sents: number of chains in the file that cross
                                   sentence boundaries
    :return sents_w_multiple_chains: number of sentences with multiple chains
    '''
    with open(file, encoding='utf8') as f:
        sent_count = 0
        tok_count = 0
        
        all_chains = set()
        sent_chains = set()  # will hold all chains encountered in the sentence
        long_chains = set()  # will prevent chains spanning more than 2 senten-
        #                      ces from being counted twice
        chains_spanning_sents = 0
        sents_w_multiple_chains = 0

        lines = f.readlines()
        # empty lines are treated as sentence boundaries, because of this, a
        # final line needs to be added in order for the final sentence to be
        # taken into consideration:
        lines.append('\n')

        for line in lines:
            line = line.rstrip().split(sep)
            # chains are contained in the last column, and have a unique number
            # captured by '\d+'
            chains = {i for i in re.findall('\d+', line[-1])}
            for chain in chains:
                # if this is the first instance of this chain in this sentence…
                if chain not in sent_chains:
                    # …but not the first in this file, nor is it known be long…
                    if chain in all_chains and chain not in long_chains:
                        chains_spanning_sents += 1  # …it spans multiple sents
                        long_chains.add(chain)  # prevent it gets counted twice
                    else:
                        all_chains.add(chain)
                    sent_chains.add(chain)
                if chain not in all_chains:
                    all_chains.add(chain)

            if len(line) == 1:  # if this is an empty line (i.e. sent boundary)
                sent_count += 1  # …count the sentence
                if len(sent_chains) > 1:  # if multiple chains were found…
                    sents_w_multiple_chains += 1  # …add to this counter
                sent_chains = set()
            else:  # if this is not an empty line…
                tok_count += 1  # …count the token

    # data is returned as a list so that it can be transposed into a pandas-
    # appropriate list of lists more easily later
    return [sent_count, tok_count, len(all_chains), 
            chains_spanning_sents, sents_w_multiple_chains]


# These two functions collect target-specific information, but it might not be
# useful to do that at the file-level. It’s in there for now in case it does
# turn out to be useful later, but it will probably be deleted eventually.

# All lines (and only those lines) that contain code related to these functions
# have been commented away twice, so that it can be detected and restored more
# easily.

# # def extract_target_info(file, query, sep='\t'):
# #     counter = 0
# #     micro_list = []
# #     with open(file, encoding='utf8') as f:
# #         for line in f.readlines():
# #             line = line.rstrip().split(sep)
# #             if query in line[-1]:
# #                 counter += 1
# #                 micro_list.append(line[-4])
# #     if counter > 0:
# #         macro_list = Counter([i[0] for i in micro_list])
# #         macro_max = [(i[0], i[1] / counter) for i in macro_list.most_common()][0]
# #         micro_list = Counter(micro_list)
# #         micro_max = [(i[0], i[1] / counter) for i in micro_list.most_common()][0]
# #         return counter, macro_max[0], macro_max[1], micro_max[0], micro_max[1]

# #     else:
# #         return counter, 'None', 0.0, 'None', 0.0


# # def collect_target_info(file, sep='\t'):
# #     output = []
# #     for query in query_list:
# #         count, macro_item, macro_share, micro_item, micro_share = extract_target_info(file, query, sep=sep)
# #         output += [count, macro_item, macro_share, micro_item, micro_share]
# #     return output


def add_info(df, columns, function):
    '''Fossil function from when there were multiple functions that extracted
       and collected information. Applies that function to create a new data-
       frame and concatenates it with some source dataframe containing file
       paths.

       :param df: the source dataframe
       :param columns: column headers of the new information
       :param function: function that takes a file path as its only required
                        input and outputs as many values as there are headers

       :return df: the source dataframe now including the newly added info
       '''
    # take all information from all files and store it in a list of lists
    info = [[function(row.file) for _, row in df.iterrows()]]

    # transpose the list of lists
    info = list(map(list, *info))

    # store it into a dataframe and concatenate it with the source dataframe
    info_df = pd.DataFrame(info, columns=columns)
    df = pd.concat([df, info_df], axis=1)
    return df


# # column_suffixes = ['_count', '_macro_item', '_micro_item', '_macro_share', '_micro_share']
# # query_list = ['B-CUE', 'I-CUE', 'B-CON', 'I-CON', 'B-SOU', 'I-SOU']
# # target_columns = [a.lower() + b for a, b in list(product(query_list, column_suffixes))]

sent_columns = ['sent_count', 'tok_count', 'chain_count',
                'chains_spanning_sents', 'sents_w_multiple_chains'] 

df = add_info(df, sent_columns, collect_sent_info)
# # df = add_info(df, target_columns, collect_target_info)

# TODO: have this point to a proper directory as well
df.to_csv('../data/output/file_level_data.tsv', sep='\t', index=False, decimal=',')
