import pandas as pd
import os
import re


def simplify_annotation(s, include_bio=True):
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
    return output if len(output) > 0 else ['O']


def preprocess_files(source_dir, target_dir, col=-1):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for subdir, dirs, files in os.walk(source_dir):
        if len(files) > 0 and 'parc' not in subdir:
            for file in files:
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf8') as f:
                    lines = []
                    for line in f.readlines():
                        line = line.rstrip().split('\t')
                        line = line[:-1] + simplify_annotation(line[-1])
                        lines.append('\t'.join(line))
                    current_folder = re.findall('(test|dev|train)', subdir)[0]
                    this_target_dir = os.path.join(target_dir, current_folder + '\\')
                    if not os.path.exists(this_target_dir):
                        os.mkdir(this_target_dir)
                    this_file = os.path.join(this_target_dir, 'p-' + file)
                    with open(this_file, 'w', encoding='utf8') as ff:
                        ff.write('\n'.join(lines))


preprocess_files('.\\data\\', '.\\preprocessed\\')
