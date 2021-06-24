import pandas as pd
import os
import re


def collect_sent_pairs(sentdir, outdir):
    for subdir, dirs, files in os.walk(sentdir):
        if len(files) > 0:
            for file in files:
                sent_ind = 0
                lines_list = []
                with open(os.path.join(subdir, file), encoding='utf8') as f:
                    lines = [line.rstrip().split('\t') for line in f.readlines()]
                    for line in lines:
                        if len(line) > 1:
                            line.append(sent_ind)
                            lines_list.append(line)
                        else:
                            sent_ind += 1
                sent_window = [0, 1]
                file_count = 0
                while sent_window[1] <= sent_ind:
                    outdir_path = os.path.join(outdir, file)
                    if not os.path.exists(outdir_path):
                        os.mkdir(outdir_path)
                    file_list = [line for line in lines_list if line[-1] in sent_window or len(line) == 1]
                    sent_window = [x + 1 for x in sent_window]
                    df = pd.DataFrame(file_list)
                    df.to_pickle(os.path.join(outdir_path, f'{str(file_count)}.pickle'))
                    file_count += 1


def fix_attributions(insts_dir):
    for subdir, dirs, files in os.walk(insts_dir):
        if len(files) > 0:
            for file in files:
                file_path = os.path.join(subdir, file)
                df = pd.read_pickle(file_path)
                atts = []
                for ind, row in df.iterrows():
                    att = re.findall('\d+', row[10])
                    if len(atts) == 0 and len(att) != 0:
                        atts.append(att[0])
                    if len(atts) > 0 and atts[0] in row[10]:
                        if len(re.findall('(CONTENT|CUE|SOURCE)', row[10])) > 0:
                            df[10][ind] = re.findall('(CONTENT|CUE|SOURCE)', row[10])[0]
                        else:
                            df[10][ind] = 'O'
                    else:
                        df[10][ind] = 'O'
                df.to_pickle(file_path)


collect_sent_pairs('.\\data\\polnear-conll\\train-conll-foreval\\', '.\\output\\')
fix_attributions('.\\output\\')