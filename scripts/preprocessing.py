import os
import re


def simplify_annotation(s, include_bio=True):
    """Simplifies the annotation into a format that only contains the BIO-tag
       and the type of AR (i.e., CUE, CONTENT, SOURCE)
    """
    cue, con, sou = '', '', ''

    if 'CUE' in s:  # Check for each type of AR
        if include_bio:  # And check for the BIO-tag if so requested
            cue = 'B-CUE' if 'B-CUE' in s else 'I-CUE'
        else:
            cue = 'CUE'
    if 'CONTENT' in s:  # Do this for the other AR types as well
        if include_bio:
            con = 'B-CONTENT' if 'B-CONTENT' in s else 'I-CONTENT'
        else:
            con = 'CONTENT'
    if 'SOURCE' in s:
        if include_bio:
            sou = 'B-SOURCE' if 'B-SOURCE' in s else 'I-SOURCE'
        else:
            sou = 'SOURCE'

    # Store only those AR types that have been found (are not empty strings)
    output = [i for i in [cue, con, sou] if len(i) > 0]
    # Return the AR types found, or an 'O' tag if no AR types were found
    return output if len(output) > 0 else ['O']


def preprocess_files(source_dir, target_dir, col=-1):
    """"Simplifies all annotations in all files, and stores the preprocessed
        files into a separate directory.
        :param source_dir: the directory currently containing the (folders to) 
                           the files
        :param target_dir: the directory where the preprocessed files should be
                           stored in
        :param col: the column the annotations are stored in, added this in
                    case some other future preprocessing step mixed the columns
                    up
    """
    if not os.path.exists(target_dir):  # Build the target dir if necessary
        os.mkdir(target_dir)
    for subdir, dirs, files in os.walk(source_dir):  # Walk through the source dir
        if len(files) > 0 and 'parc' not in subdir:  # if in a folder that 1) contains files and 2) is not parc…
            for file in files:  # …go through all files
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf8') as f:
                    lines = []  # use this to store all preprocessed lines in
                    for line in f.readlines():
                        line = line.rstrip().split('\t')  # split each line into columns
                        line = line[:-1] + simplify_annotation(line[-1])  # simplify the final column
                        lines.append('\t'.join(line))  # and stitch them back together
                    current_folder = re.findall('(test|dev|train)', subdir)[0]  # check whether in the dev, train, or test set
                    this_target_dir = os.path.join(target_dir, current_folder + '\\')  # name the folder accordingly
                    if not os.path.exists(this_target_dir):
                        os.mkdir(this_target_dir)  # and build it if that hadn’t already been done
                    this_file = os.path.join(this_target_dir, 'p-' + file)  # add a p- prefix to preprocessed files
                    with open(this_file, 'w', encoding='utf8') as ff:
                        ff.write('\n'.join(lines))  # and write the lines into a file


preprocess_files('..\\data\\data\\', '..\\data\\preprocessed\\')
