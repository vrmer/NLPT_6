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