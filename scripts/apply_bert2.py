import torch
import transformers as tf

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def prepare_segment_numbers(tokens, max_len):
    """

    :param tokens:
    :param max_len:
    :return:
    """
    prepared_segment_numbers = np.concatenate((['[CLS]'], tokens, ['[SEP]']))
    padded_prepared_segment_numbers = np.concatenate(
        (prepared_segment_numbers, ['[PAD]'] * (max_len - len(prepared_segment_numbers)))
    )
    return padded_prepared_segment_numbers


# path = '../data/parc30-conll/train-conll-foreval/wsj_0069.xml.conll.features.foreval'
path = '../data/parc30-conll/train-conll-foreval/wsj_0017.xml.conll.features.foreval'

df = pd.read_csv(path, delimiter='\t', header=None)

sentence_groups = df.groupby(1)[5]

string_sentences = []
list_sentences = []

# Extract sentences in string and list format
for idx, sentence in sentence_groups:
    string_sentence = ' '.join(sentence.values)
    string_sentences.append(string_sentence)

    list_sentences.append(sentence.values)

# Initialize BERT
model_class, tokenizer_class, pretrained_weights = (
    tf.DistilBertModel,
    tf.DistilBertTokenizer,
    'distilbert-base-uncased'
)

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized_sentences = []

for string_sentence in string_sentences:

    tokens = tokenizer.encode(string_sentence, add_special_tokens=True)
    tokenized_sentences.append(tokens)

# Padding time
max_len = 0

for sent in tokenized_sentences:
    if len(sent) > max_len:
        max_len = len(sent)

padded_sentences = np.array(
    [sent + [0] * (max_len - len(sent))
     for sent in tokenized_sentences]
)

attention_mask = np.where(padded_sentences != 0, 1, 0)


decoded_padded_sentences = [
    tokenizer.convert_ids_to_tokens(sentence)
    for sentence in padded_sentences
]

padded_segment_numbers = []
segment_numbers = []

for sent in list_sentences:

    segment_number = [
        len(tokenizer.tokenize(word))
        for word in sent
    ]

    segment_numbers.append(segment_number)

    padded_segment_number = prepare_segment_numbers(segment_number, max_len)

    padded_segment_numbers.append(padded_segment_number)

# print(padded_segment_numbers)

alignment = []

for segs, sent in zip(segment_numbers, list_sentences):
    sent_alignment = []
    for seg, word in zip(segs, sent):
        sent_alignment.append(word)
        if seg != 1:
            sent_alignment.extend(['[PART]'] * (seg - 1))
    padded_sent_alignment = prepare_segment_numbers(sent_alignment, max_len)
    alignment.append(padded_sent_alignment)

print(alignment)

# for al in alignment:
#     print(prepare_segment_numbers(al, max_len))

# Training DistilBERT
# print(padded_sentences)
input_ids = torch.tensor(padded_sentences)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

encoded_sentences = [
    last_hidden_states[0][i, :, :]
    for i in range(len(padded_sentences))
]

# print(len(encoded_sentences[0]))

to_output = []
clss = []

for idx, (enc_sent, als) in enumerate(zip(encoded_sentences, alignment)):

    to_add = []
    sentence_tokens = []

    for enc, al in zip(enc_sent, als):
        # print(al)

        if al == '[CLS]':
            clss.append(enc)

        elif al not in ['[SEP]', '[PAD]', '[PART]']:

            # if als[idx+1] != '[PART]':

                # if to_add:
                #     new_enc = torch.mean(torch.stack(to_add))
                #     sentence_tokens.append(new_enc)
                #     to_add = []

            sentence_tokens.append(enc)
            #
            # else:
            #     to_add = [enc]

        # elif al == '[PART]':
        #     to_add.append(enc)

    to_output.append(sentence_tokens)

# print(len(to_output[1]))
for o in to_output:
    print(len(o))

# print(to_output)
# print(clss)

# print(segment_numbers)

# print(padded_segment_numbers)
# for segs, sent in zip(wordpiece_segment_counts, list_sentences):
#     for seg, word in zip(segs, sent):
#         alignment.append(word)
#         if seg != 1:
#             alignment.extend(['PART'] * (seg - 1))

# for sent in padded_segment_numbers:


    # sent.insert(0, '[CLS')
    # new_sent = np.concatenate((['[CLS]'], sent, ['[SEP]']))
    # new_sent = np.insert(sent, 0, '[CLS]')
    # new_sent = np.insert(new_sent, -1, '[SEP]')
    # print(new_sent)
#     new_sent = ['CLS'] + sent + ['SEP']
#     print(new_sent)

# padded_list_sentences = [
#     sent + ['[PAD]'] * (max_len - len(sent))
#     for sent in list_sentences
# ]
#
# for sent in list_sentences:
#     print(sent)

# print(padded_list_sentences)

# Tokenize sentences and keep track of the number of segments tokens are split into

# wordpiece_segment_counts = []
#
# for string_sentence in string_sentences:
#
#     tokens = tokenizer.encode(string_sentence, add_special_tokens=True)
#     tokenized_sentences.append(tokens)
#
# # Padding time
# max_len = 0
#
# for i in tokenized_sentences:
#     if len(i) > max_len:
#         max_len = len(i)
#
# padded = np.array(
#     [i + [0]*(max_len - len(i)) for i in tokenized_sentences]
# )
#
# # Create the attention mask so we know what to ignore
# attention_mask = np.where(padded != 0, 1, 0)
#
# # Training DistilBERT
# input_ids = torch.tensor(padded)
# attention_mask = torch.tensor(attention_mask)
#
# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
#
# for sentence in padded:
#     sentence_segments = []
#
#     for word in sentence:
#         tokenized_word = tokenizer.tokenize(word)
#         sentence_segments.append(len(tokenized_word))
#
#     wordpiece_segment_counts.append(sentence_segments)
#
# decoded_sentences = [
#     tokenizer.convert_ids_to_tokens(sent)
#     for sent in tokenized_sentences
# ]
#
# print(wordpiece_segment_counts)
#
# alignment = []
#
# for segs, sent in zip(wordpiece_segment_counts, list_sentences):
#     for seg, word in zip(segs, sent):
#         alignment.append(word)
#         if seg != 1:
#             alignment.extend(['PART'] * (seg - 1))
#
# print(alignment)
#
# # Postprocessing
# article_alignments = []
#
# for sentence in decoded_sentences:
#
#     sentence_alignment = []
#
#     for dec in sentence:
#
#         if dec == '[CLS]':
#             sentence_alignment.append('CLS')
#         elif dec not in ['[SEP]', '[PAD]']:
#             sentence_alignment.append('TOKEN')
#         else:
#             sentence_alignment.append('JUNK')
#
#     article_alignments.append(sentence_alignment)
#
# # print(article_alignments)
#
# # print(decoded_sentences[0])
#
# to_output = []
# clss = []
# last_index = 0
#
# for idx, (number, word) in enumerate(zip(wordpiece_segment_counts[0], decoded_sentences[0])):
#     if not decoded_sentences[0][idx] == '[CLS]':
#         if number == 1:
#             word = decoded_sentences[0][idx]
#             print(word)
#         else:
#             word = decoded_sentences[0][idx:idx+number]
#         # print(word)
#
# # for idx, number in enumerate(token_segments[0]):
# #
# #     if last_index is False:
# #         real_idx = idx
# #     else:
# #         real_idx = last_index
# #
# #     if number == 1:
# #         if article_alignments[0][real_idx] == 'CLS':
# #             clss.append(decoded_sentences[0][real_idx])
# #         elif article_alignments[0][real_idx] != 'JUNK':
# #             to_output.append(decoded_sentences[0][real_idx])
# #
# #         last_index += 1
# #
# #     else:
# #         last_index = real_idx + number + 1
# #         token = decoded_sentences[0][real_idx + 1:last_index]
# #         for item in decoded_sentences[0][real_idx + 1:last_index]:
# #             print(item)
# #         token = ''.join(token)
# #         to_output.append(token)
# #
# # # for o in to_output:
# # #     print(o)
# # print(to_output)
#
#
# # for idx, (dec, al) in enumerate(zip(decoded_sentences[0], article_alignments[0])):
# #     if al == 'CLS':
# #         clss.append(dec)
# #     elif al != 'JUNK':
# #         try:
# #             n_segments = token_segments[0][idx]
# #             print(n_segments)
# #         except IndexError:
# #             print('Out of index')
#
#         # if n_segments == 1:
#         #     to_output.append(dec)
#         #
#         # else:
#         #     token = [
#         #         segment for segment in token_segments[idx:idx+n_segments]
#         #     ]
#         #     print(''.join(token))
#     #         to_output.append(dec)
#             # else:
#             #     token = []
#             #     counter += (idx + seg + 1)
#             #     for i in range(idx, (idx + seg + 1)):
#             #         token.append(decoded_sentences[0][i])
#             #     join_token = ''.join(token)
#             #     to_output.append(join_token)
    # else:
    #     counter -= 1


# print()
# print()
# print(clss)
# print(to_output)

