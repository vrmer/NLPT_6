"""
See inspiration from https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb#scrollTo=9T_1GbrdWxiH
"""


def create_token_alignment(encoded_sentence, decoded_sentence):
    """
    This function keeps only the first segment of the result of BERT tokenization,
    allowing sequence labelling to take place via the use of BERT and preventing
    alignment errors.

    :param encoded_sentence: token ids as created by the BERT tokenizer
    :param decoded_sentence: tokens stringified using the BERT tokenizer convert_ids_to_tokens method
    :return: a list of the original input tokens and a list of token ids standing for the first segment for each token
    """
    # input_sentence = 'whatever'
    # encoded_sentence = tokenizer.encode(input_sentence, is_split_into_words=True, add_special_tokens=True)
    # decoded_sentence = tokenizer.convert_ids_to_tokens(encoded_sentence)

    original_tokens = []
    cleaned_token_ids = []

    for idx, (enc_token, dec_token) in enumerate(zip(encoded_sentence, decoded_sentence)):
        if dec_token.startswith('##'):
            # try:
            original_tokens.remove(decoded_sentence[idx-1])
            # except:
            new_token = decoded_sentence[idx-1] + decoded_sentence[idx].lstrip('##')
            original_tokens.append(new_token)
        else:
            original_tokens.append(dec_token)
            cleaned_token_ids.append(enc_token)

    return original_tokens, cleaned_token_ids
