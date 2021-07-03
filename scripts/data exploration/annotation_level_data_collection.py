import os
import pandas as pd
import re


def simplify_target(s, include_bio=False):
    cue, con, sou = '', '', ''

    if 'CUE' in s:
        if include_bio:
            cue = 'B-CUE' if 'B-CUE' in s else 'I-CUE'
        else:
            cue = 'CUE'
    if 'CONTENT' in s:
        if include_bio:
            con = 'B-CONTENT' if 'B-CONTENT' in s else 'I-CONTENT'
        else:
            con = 'CONTENT'
    if 'SOURCE' in s:
        if include_bio:
            sou = 'B-SOURCE' if 'B-SOURCE' in s else 'I-SOURCE'
        else:
            sou = 'SOURCE'
    output = [i for i in [cue, con, sou] if len(i) > 0]
    return output if len(output) > 0 else 'O'


file_location = '../../data/output/annotation_level_data.tsv'

if not os.path.exists(file_location):
    file_list = []
    for subdir, dirs, files in os.walk('.\\data\\'):
        if len(files) > 0 and 'test' not in subdir:
            file_list += [os.path.join(subdir, file) for file in files]

    all_rows = []
    for file in file_list:
        with open(file, encoding='utf8') as f:
            lines = f.readlines()
            for row in lines:
                if type(row) == str:
                    row = row.rstrip().split('\t')
                if len(row) > 1:
                    row = row[:-1] + [simplify_target(row[-1])[0]]
                    if len(simplify_target(row[-1])) > 1:
                        for i in simplify_target(row[-1])[1:]:
                            lines.append(row[:-1] + [i])
                    row = row[:8] + [row[7][0]] + row[8:-2] + [row[-1]]
                    row = [file] + row[5:]
                    all_rows.append(row)

    all_rows = list(map(list, *[all_rows]))
    columns = ['file', 'token', 'lemma', 'pos_micro', 'pos_macro', 'dep_label', 'target']
    df = pd.DataFrame(all_rows, columns=columns)
    df['source'] = df.file.apply(lambda x: 'PARC' if 'parc' in x else 'PoLNeAR')
    df['type'] = df.file.apply(lambda x: re.findall('(train|dev)', x)[0])

    df.to_csv(file_location, sep='\t', decimal=',')
else:
    df = pd.read_csv(file_location, sep='\t')
