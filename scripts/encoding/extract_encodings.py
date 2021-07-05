import glob

from tqdm import tqdm
from encoding.bert import process_document as process_document

# Defining paths to data
parc30 = glob.glob('../../data/corpora/parc30-conll/**/**')
polnear = glob.glob('../../data/corpora/polnear-conll/**/**')

all_paths = parc30 + polnear

# Running the code from BERT script for parc and polnear corpus
with tqdm(total=len(all_paths), desc='Encoding articles... ') as pbar:

    for path in parc30:

        corpus = 'parc30-conll'

        process_document(path, corpus)

        pbar.update(1)

    for path in polnear:

        corpus = 'polnear-conll'

        process_document(path, corpus)

        pbar.update(1)
