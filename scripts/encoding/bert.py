import joblib
import os
import torch
import transformers as tf

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


def prepare_segment_numbers(tokens, max_len):
    """
    Append cls, sep and pad tokens.
    :tokens: the tokens in the article for which to generate the representations
    :max len: the maximum length for padding
    """
    prepared_segment_numbers = np.concatenate((['[CLS]'], tokens, ['[SEP]']))
    padded_prepared_segment_numbers = np.concatenate(
        (prepared_segment_numbers, ['[PAD]'] * (max_len - len(prepared_segment_numbers)))
    )
    return padded_prepared_segment_numbers


def extract_sentences_from_df(path):
    """
    Get string and list representation of the sentences
    in an input document.
    :path: path to document
    """
    string_sentences = []
    list_sentences = []

    df = pd.read_csv(path, delimiter='\t', header=None)

    sentence_groups = df.groupby(1)[5]

    # Extract sentences in string and list format
    for idx, sentence in sentence_groups:
        string_sentence = ' '.join(sentence.values)
        string_sentences.append(string_sentence)

        list_sentences.append(sentence.values)

    return string_sentences, list_sentences


def initialize_bert():
    """
    Initialize DistilBERT and its tokenizer,
    returns the tokenizer and the model.
    """
    # Initializing model and tokenizer
    model_class, tokenizer_class, pretrained_weights = (
        tf.DistilBertModel,
        tf.DistilBertTokenizer,
        'distilbert-base-uncased'
    )

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    return tokenizer, model


def tokenize_sentences(string_sentences, tokenizer):
    """
    Tokenizes the string input sentences.
    """
    tokenized_sentences = []

    # Tokenizing the data
    for string_sentence in string_sentences:
        tokens = tokenizer.encode(string_sentence, add_special_tokens=True)
        tokenized_sentences.append(tokens)

    return tokenized_sentences


def padding_time(tokenized_sentences):
    """
    Carries out padding on the tokenized sentences,
    returning the padded sentences, the attention mask,
    and the max length of the sentences in the input document.
    """
    max_len = 0

    for sent in tokenized_sentences:
        if len(sent) > max_len:
            max_len = len(sent)

    # Padding sentences to make the input of same length for every sequence
    padded_sentences = np.array(
        [sent + [0] * (max_len - len(sent))
         for sent in tokenized_sentences]
    )

    # Setting the attention mask
    attention_mask = np.where(padded_sentences != 0, 1, 0)

    return padded_sentences, attention_mask, max_len


def extract_token_segment_numbers(list_sentences, tokenizer):
    """
    Returns how many segments a single token is split
    into by the BERT tokenizer.
    """
    segment_numbers = []

    for sent in list_sentences:

        segment_number = [
            len(tokenizer.tokenize(word))
            for word in sent
        ]

        segment_numbers.append(segment_number)

    return segment_numbers


def create_alignment_list(segment_numbers, list_sentences, max_len):
    """
    Creates an alignment list that allows the
    extraction of token encodings aligning with
    the input document.
    """
    alignment = []

    # Zipping through list and segment N to ensure alignment
    for segment_number, list_sentence in zip(segment_numbers, list_sentences):

        sentence_alignment = []

        for number, word in zip(segment_number, list_sentence):
            sentence_alignment.append(word)

            # Adding PART tokens for follwing multiword tokens in input
            if number != 1:
                sentence_alignment.extend(['[PART]'] * (number - 1))

        padded_sentence_alignment = prepare_segment_numbers(sentence_alignment, max_len)

        alignment.append(padded_sentence_alignment)

    return alignment


def encode_sentences(padded_sentences, attention_mask, model):
    """
    Create BERT encodings of the input sentences.
    """
    input_ids = torch.tensor(padded_sentences)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    encoded_sentences = [
        last_hidden_states[0][i, :, :]
        for i in range(len(padded_sentences))
    ]

    return encoded_sentences


def decrease_dimensionality(input_encoding, pca, first_index=True):
    """
    This function decreases dimensionality of the BERT output.
    """
    prepared_encoding = input_encoding.numpy().reshape(32, 24)

    # Decreasing dimensionality from 768 BERT tokens to 24, because of memory limitations
    if first_index is True:
        pca_encoding = pca.fit_transform(prepared_encoding)
    else:
        pca_encoding = pca.transform(prepared_encoding)

    return pca_encoding


def extract_token_level_encodings(encoded_sentences, alignment, corpus, path, pca):
    """
    Return the token level encodings and the sentence level encodings.
    """
    filename = os.path.basename(path)
    dir_name = os.path.basename(
        os.path.dirname(path)
    )

    # Creating output folders
    os.mkdir(f'../data/encodings/{corpus}/{dir_name}/{filename}')

    for idx, (encoded_sentence, sentence_alignment) in enumerate(zip(encoded_sentences, alignment)):

        sentence_tokens = []

        output_name = f'../data/encodings/{corpus}/{dir_name}/{filename}/{idx}.sav'

        for encoded_token, al in zip(encoded_sentence, sentence_alignment):

            if al not in ['[SEP]', '[PAD]', '[PART]']:

                if idx == 0:
                    reduced_tokens = decrease_dimensionality(encoded_token, pca)
                else:
                    reduced_tokens = decrease_dimensionality(encoded_token, pca, first_index=False)

                sentence_tokens.append(reduced_tokens)

        joblib.dump(sentence_tokens, output_name, compress=True)


tokenizer, model = initialize_bert()
pca = PCA(n_components=24, svd_solver='auto')


def process_document(path, corpus):
    """
    This function takes an input path, and it creates
    encodings for each sentence and each token in the sentences.
    """
    string_sentences, list_sentences = extract_sentences_from_df(path)
    tokenized_sentences = tokenize_sentences(string_sentences, tokenizer)
    padded_sentences, attention_mask, max_len = padding_time(tokenized_sentences)
    segment_numbers = extract_token_segment_numbers(list_sentences, tokenizer)
    alignment = create_alignment_list(segment_numbers, list_sentences, max_len)
    encoded_sentences = encode_sentences(padded_sentences, attention_mask, model)

    extract_token_level_encodings(encoded_sentences, alignment, corpus, path, pca)
