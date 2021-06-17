import os
import glob
import pickle
import gc

from tqdm import tqdm
from encoding.bert import process_document as process_document


def encode_output_document(path, corpus):
    """
    This function encodes an input document,
    and outputs it in the appropriate folder
    in a pickle format.

    :param path: input path
    :param corpus: which corpus the input file belongs to
    :return: None
    """
    filename = os.path.basename(path)
    dir_name = os.path.basename(
        os.path.dirname(path)
    )

    output_name = f'../data/encodings/{corpus}/{dir_name}/{filename}.pkl'

    token_encodings, sentence_encodings = process_document(path)

    article_representation = {
        'tokens_per_sentence': token_encodings,
        'sentences': sentence_encodings
    }

    with open(output_name, 'wb') as outfile:
        pickle.dump(article_representation, outfile)

    del article_representation
    del token_encodings
    del sentence_encodings
    gc.collect()


parc30 = glob.glob('../data/corpora/parc30-conll/**/**')
polnear = glob.glob('../data/corpora/polnear-conll/**/**')

all_paths = parc30 + polnear

with tqdm(total=len(all_paths), desc='Encoding articles... ') as pbar:

    for path in parc30:

        corpus = 'parc30-conll'

        encode_output_document(path, corpus)

        pbar.update(1)

    for path in polnear:

        corpus = 'polnear-conll'

        encode_output_document(path, corpus)

        pbar.update(1)
