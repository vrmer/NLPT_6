import torch
import transformers as tf

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression


def embedding_extraction(path):
    '''

    :param path: path to file for processing
    :return:
    '''

    # Read the input file
    df = pd.read_csv(path, delimiter='\t', header=None)

    # Merging the tokens into sentences
    sentences = df.groupby(1)[5]
    sents = []
    sent_token_list = []
    segmented_numbers = []

    for idx, sentence in sentences:
        sent_token_list.append(sentence.values)
        sent = ' '.join(sentence.values)
        sents.append(sent)

    print(sent_token_list)

    # load DistilBERT model
    model_class, tokenizer_class, pretrained_weights = (
        tf.DistilBertModel,
        tf.DistilBertTokenizer,
        'distilbert-base-uncased'
    )

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    for sent in sent_token_list:
        sent_numbers = []
        for token in sent:
            sent_numbers.append(len(tokenizer.tokenize(token)))
        segmented_numbers.append(sent_numbers)

    print(segmented_numbers)
    for seg in segmented_numbers:
        print(len(seg))

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
    sentences_encoded = []
    sentences_decoded = []

    for i in range(len(sents)):
        # sentence_encoded = last_hidden_states[0][i, :, :]
        sentence_encoded = last_hidden_states[0][i, :, :]
        sentence_decoded = tokenizer.convert_ids_to_tokens(padded[i])

        sentences_encoded.append(sentence_encoded)
        sentences_decoded.append(sentence_decoded)

    print(sentences_decoded)

    # Create an alignment list where important tokens are marked with CLS, TOKEN, or PART, rest is marked as JUNK
    alignment_list = []

    for sentence in sentences_decoded:

        candidate_list = []

        for dec in sentence:

            if dec == '[CLS]':
                candidate_list.append('CLS')
            elif dec.startswith('##'):
                candidate_list.append('PART')
            elif dec not in ['[SEP]', '[PAD]']:
                candidate_list.append('TOKEN')
            else:
                candidate_list.append('JUNK')

        alignment_list.append(candidate_list)

    # Create sentence representation in output list for each token, while gathering CLSs in their own list
    output_encodings = []
    clss = []

    for enc_sent, dec_sent, als in zip(sentences_encoded, sentences_decoded, alignment_list):

        sentence_tokens = []
        token_reassembly = []

        for idx, (enc, dec, al) in enumerate(zip(enc_sent, dec_sent, als)):

            # Collect CLSs
            if al == 'CLS':
                clss.append(enc)

            # Exclude 'JUNK'
            elif al != 'JUNK':

                if al == 'TOKEN':

                    # Average the subword tokens in token_reassembly if it exists
                    if token_reassembly:
                        new_enc = torch.mean(torch.stack(token_reassembly))
                        sentence_tokens.append(new_enc)
                        token_reassembly = []

                    # If token is standalone, not subword tokenized, just add its representation to the output list
                    if als[idx + 1] not in ['PART', 'JUNK']:
                        sentence_tokens.append(enc)

                    # Else just add it as a candidate to 'token_reassembly'
                    else:
                        token_reassembly.append(enc)

                # If a token is PART, add it to 'token_reassembly'
                elif al == 'PART':
                    token_reassembly.append(enc)

        output_encodings.append(sentence_tokens)

    # print(output_encodings)
    # for item in output_encodings:
        # print(len(item))
    # print()
    # for item in alignment_list:
    #     print(len(item))
    #     print(item)
    # for dec in sentences_decoded:
    #     print(dec)
    # print(clss)
    return output_encodings, tokenizer, padded, clss


# path = '../../parc30-conll/train-conll-foreval/wsj_0012.xml.conll.features.foreval'
path = '../data/parc30-conll/train-conll-foreval/wsj_0069.xml.conll.features.foreval'
output_encodings, tokenizer, padded, clss = embedding_extraction(path)

# print(len(output_encodings[0]))

# print(tokenizer.char_to_token('a'))



# tokenized_word = tokenizer.tokenize('Myrthe hates unconsiderateness')
# n_subwords = len(tokenized_word)
# print(tokenized_word, n_subwords)
