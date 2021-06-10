import torch
import transformers as tf

import numpy as np
import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'D:\VU Amsterdam\NLPT_6\data\parc30-conll\train-conll-foreval\wsj_0012.xml.conll.features.foreval',
                 delimiter='\t', header=None)

# Merging the tokens into sentences
sentences = df.groupby(1)[5]
sents = []
for idx, sentence in sentences:
    sent = ' '.join(sentence.values)
    # print(idx)
    # print(sent)
    sents.append(sent)

# load DistilBERT
model_class, tokenizer_class, pretrained_weights = (
    tf.DistilBertModel,
    tf.DistilBertTokenizer,
    'distilbert-base-uncased'
)

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Tokenize sentences
tokenized = []
for sent in sents:
    tokens = tokenizer.encode(sent, add_special_tokens=True)
    tokenized.append(tokens)

# Padding time
max_len = 0

for i in tokenized:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array(
    [i + [0]*(max_len - len(i)) for i in tokenized]
)

# Create the attention mask so we know what to ignore
attention_mask = np.where(padded != 0, 1, 0)

# Training DistilBERT
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# last_hidden_states format: [0][sentence number, token position, hidden unit outputs]
# first and last tokens are [CLS] and [SEP]
# multiple [PAD] tokens might appear after [SEP]
# one token has 768 dimensions of embeddings
first_sentence_encoded = last_hidden_states[0][0, :, :]
first_sentence_decoded = tokenizer.convert_ids_to_tokens(padded[0])

# Create an alignment list where important tokens are marked with CLS, TOKEN, or PART, rest is marked as JUNK
alignment_list = []

for dec in first_sentence_decoded:
    if dec == '[CLS]':
        alignment_list.append('CLS')
    elif dec.startswith('##'):
        alignment_list.append('PART')
    elif dec not in ['[SEP]', '[PAD]']:
        alignment_list.append('TOKEN')
    else:
        alignment_list.append('JUNK')

# Create sentence representation in output list for each token, while gathering CLSs in their own list
output = []
clss = []

token_reassembly = []

for idx, (enc, dec, al) in enumerate(zip(first_sentence_encoded, first_sentence_decoded, alignment_list)):
    # Collect CLSs
    if al == 'CLS':
        clss.append(enc)

    # Exclude 'JUNK'
    elif al != 'JUNK':

        if al == 'TOKEN':

            # Average the subword tokens in token_reassembly if it exists
            if token_reassembly:
                new_enc = torch.mean(torch.stack(token_reassembly))
                output.append(new_enc)
                token_reassembly = []

            # If token is standalone, not subword tokenized, just add its representation to the output list
            if alignment_list[idx + 1] != 'PART':
                output.append(enc)

            # Else just add it as a candidate to 'token_reassembly'
            else:
                token_reassembly.append(enc)

        # If a token is PART, add it to 'token_reassembly'
        elif al == 'PART':
            token_reassembly.append(enc)

print(len(output))
