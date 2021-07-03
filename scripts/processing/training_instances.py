import pandas as pd
import os
import re
import warnings
warnings.filterwarnings("ignore")


def collect_sents(sentdir, outdir):
    """
    Collects sentences from the conll files and stores them in a new directory.
    """
    for subdir, dirs, files in os.walk(sentdir):
        if len(files) > 0:
            for file in files:
                sent_ind = 0  # set a sentence index
                lines_list = []
                with open(os.path.join(subdir, file), encoding='utf8') as f:
                    # create a list of lists containing all split rows in the file
                    lines = [line.rstrip().split('\t') for line in f.readlines()]
                    for line in lines:
                        if len(line) > 1:  # if it is not an empty line
                            line.append(sent_ind)  # add the sentence index to the row
                            lines_list.append(line)  # and add the line to the lines list
                        else:  # if it is an empty line (i.e., end of a sentence)
                            sent_ind += 1  # increase the sentence index
                file_count = 0  # set a file counter
                for i in range(sent_ind):  # iterate through the number of sentences found in the file
                    outdir_path = os.path.join(outdir, file)  # name a folder after the file
                    if not os.path.exists(outdir_path):
                        os.mkdir(outdir_path)
                    file_list = [line for line in lines_list if line[-1] == i]  # add only those lines that correspond to the current sentence
                    df = pd.DataFrame(file_list)  # turn them into a df and save it to a file
                    df.to_pickle(os.path.join(outdir_path, f'{file_count}.pickle'))
                    file_count += 1


def fix_attributions(insts_dir, verbose=False):
    """Simplify attributions so that it is one of:
        -SOURCE
        -CUE
        -CONTENT
        -O
    """
    for subdir, dirs, files in os.walk(insts_dir):
        if len(files) > 0:
            counter = 0
            for file in files:
                file_path = os.path.join(subdir, file)
                df = pd.read_pickle(file_path)
                atts = []
                for idx, row in df.iterrows():
                    att = re.findall('\d+', row[10])
                    if len(atts) == 0 and len(att) != 0:
                        atts.append(att[0])
                    if len(atts) > 0 and atts[0] in row[10]:
                        if len(re.findall('(CONTENT|CUE|SOURCE)', row[10])) > 0:
                            df[10][idx] = re.findall('(CONTENT|CUE|SOURCE)', row[10])[0]
                        else:
                            df[10][idx] = 'O'
                    else:
                        df[10][idx] = 'O'
                df.to_pickle(file_path)
                counter += 1
                if counter % 100 == 0 and verbose:
                    print(file_path, f"{str(counter)}/{str(len(files))}")


collect_sents('.\\data\\polnear-conll\\train-conll-foreval\\', '.\\output\\train-conll-foreval\\')
fix_attributions('.\\output\\train-conll-foreval\\')
