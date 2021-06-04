import glob
import pickle
import pandas as pd

from collections import Counter, defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt

plt.style.use('seaborn')

headers = [
    'file_name',
    'sent_num',
    'token_num_doc',
    'token_num_sent',
    'begin_end_offset',
    'token',
    'lemma',
    'pos',
    'dep_label',
    'dep_head',
    'attr_labels'
]

conll_kwargs = {
    'delimiter': '\t',
    'encoding': 'utf8',
    'names': headers
}

label_dict = {
    'sources': [
        'SOURCE',
        'B-SOURCE',
        'I-SOURCE'
    ],
    'contents': [
        'CONTENT',
        'B-CONTENT',
        'I-CONTENT'
    ],
    'cues': [
        'CUE',
        'B-CUE',
        'I-CUE'
    ]
}


def extract_label(data, label):
    """
    Given a label ('SOURCE', 'CONTENT', OR 'CUE'),
    this script filters for columns that contain
    a token annotated for that label.

    :param data: input dataframe
    :param label: target label ('SOURCE', 'CONTENT', 'CUE')
    :return: a filtered dataframe only containing columns
    where the token is annotated for the target label
    """
    try:
        return data.query(f'attr_labels.str.contains("{label}")')

    # If there is no source, content or cue
    except ValueError:
        return None

    # If there is no attribution at all
    except AttributeError:
        return None


def extract_examples(file_path,
                     target_label='sources',
                     show_errors=False):
    """
    This function takes examples of sources, cues, or content
    from the articles and returns the examples in a list.

    :param file_path: input filepath
    :param target_label: sources, contents or cues
    :param show_errors: if True, error log is printed containing words that are cues or sources in multiple chains
    :return:
    """
    dataframe = pd.read_csv(
        file_path,
        **conll_kwargs
    )

    filtered_dataframe = extract_label(
        dataframe,
        label_dict[target_label][0]
    )

    examples = []
    example = []
    errors = []

    if filtered_dataframe is not None:

        for idx, row in filtered_dataframe.iterrows():
                labels = row.attr_labels.split()

                for label in labels:
                    counter = 0
                    if label.startswith(label_dict[target_label][1]):

                        if example:

                            examples.append(
                                ' '.join(example)
                            )

                            example = []

                        counter += 1
                        example.append(row.token)

                        if counter > 1:
                            print(f'WARNING: Multiple {label_dict[target_label][1]} in one row!')
                            errors.append(row.token)

                    elif label.startswith(label_dict[target_label][2]):
                        example.append(row.token)

                    else:
                        continue

    else:
        examples = []

    if show_errors is True:
        print(errors)

    return examples


all_cues = []
all_sources = []

all_files = glob.glob('../data/**/**/**')

non_test_files = [
    filename for filename in all_files
    if 'test' not in filename
]

# Looping through the files inside all folders that are not part of the test set
# with tqdm(total=len(non_test_files), desc='Looping through articles: ') as pbar:
#
#     for file_path in non_test_files:
#         # print(file_path)
#         cue_examples = extract_examples(file_path,
#                                         target_label='cues')
#         source_examples = extract_examples(file_path,
#                                            target_label='sources')
#
#         for cue, source in zip(cue_examples, source_examples):
#             all_cues.append(cue)
#             all_sources.append(source)
#
#         pbar.update(1)


# counted_cues = Counter(all_cues)
# counted_sources = Counter(all_sources)
#
# with open('counted_cues.pkl', 'wb') as outfile:
#     pickle.dump(counted_cues, outfile)
#
# with open('counted_sources.pkl', 'wb') as outfile:
#     pickle.dump(counted_sources, outfile)

with open('counted_cues.pkl', 'rb') as infile:
    counted_cues = pickle.load(infile)

with open('counted_sources.pkl', 'rb') as infile:
    counted_sources = pickle.load(infile)

cue_dict = defaultdict(int)
source_dict = defaultdict(int)

for cue, freq in counted_cues.items():
    cue_dict[freq] += 1

for source, freq in counted_sources.items():
    source_dict[freq] += 1

print(sorted(cue_dict.items(), key=lambda x: x[1], reverse=True))
print(source_dict)

# cues_df = pd.DataFrame(counted_cues, index=counted_cues.keys())
# cues_df.drop(cues_df.columns[1:], inplace=True)

# cues_df.plot(kind='bar')

# plt.show()

# with open('../data/output/data_exploration/all_cues.txt', 'w', encoding='utf8') as outfile:
#     for cue in sorted(all_cues):
#         outfile.write(cue)
#         outfile.write('\n')
#
# with open('../data/output/data_exploration/all_sources.txt', 'w', encoding='utf8') as outfile:
#     for source in sorted(all_sources):
#         outfile.write(source)
#         outfile.write('\n')

# with open('../data/output/data_exploration/more_frequent_cues.txt', 'w', encoding='utf8') as outfile:
#     for cue, freq in counted_cues.items():
#         # if freq > 1:
#         if freq > 5:
#             outfile.write(cue)
#             outfile.write('\n')
#
# with open('../data/output/data_exploration/more_frequent_sources.txt', 'w', encoding='utf8') as outfile:
#     for source, freq in counted_sources.items():
#         # if freq > 1:
#         if freq > 5:
#             outfile.write(source)
#             outfile.write('\n')
