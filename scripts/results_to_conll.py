import pandas as pd
import os
import joblib
import warnings
warnings.filterwarnings("ignore")


def add_biotags(df, number, column=10):
    """
    Adds BIO-tags and IDs to predictions in a given df. Since it works in
    tandem with another function that adds different number IDs to different
    chains, this function also updates the number ID if a chain is found.

    :param df: the df whose tags are to be edited
    :param number: the number ID that is to be added to the predictions
    """
    attr_set = set()  # used to check whether a B or I tag is to be added
    chains_found = False  # used to check whether a chain has been found

    for ind, row in df.iterrows():  # iterate through the rows of the df
        attr = row[column]  # get the prediction from the row
        if attr != 'O':  # if it is something other than O
            if attr in attr_set:  # add an I if another one had been found
                df[column][ind] = 'I-' + attr + f'-{number + 1}'
            else:  # if not
                attr_set.add(attr)  # add it to attr_set
                # add a B tag
                df[column][ind] = 'B-' + attr + f'-{number + 1}'
                if not chains_found:  # and note that a chain was found
                    chains_found = True
    if chains_found:  # if a chain was found
        number += 1   # update number so that different chains get different IDs
    return df, number


def find_number_in_tag(tag):
    """ 
    Get the number from a tag without having to use regex
    """
    numbers = {str(i) for i in range(10)}  # get a set of all digits as strings
    while tag[0] not in numbers:  # check if the first char is a digit
        if len(tag) > 1:  # if it is not the only remaining char in the string
            tag = tag[1:]  # cut it off
        else:  # if it is the only remanining char in the string
            tag = None  # return None as there is no digit
            break  # and break the while-loop
    return tag


def subfunction_subcolumns(label, max_number):
    """
    Subfunction to the subcolumn function, adds subcolumns to a given label
    
    :param label: the label around which the subcolumns should be added
    :param max_number: the number of subcolumns that should be added
    """
    columns = []
    for n in range(max_number + 1):
        # if the current iteration number corresponds to the label number
        if find_number_in_tag(label) == str(n + 1):
            columns.append(label)  # add the label to the column list
        elif n + 1 <= max_number:  # if not
            columns.append('_')  # add an underscore
    return ' '.join(columns)  # join them together using spaces


def add_subcolumns(df):
    """
    Adds subcolumns to a given df
    """
    # get all number IDs that occur in the df
    number_list = df[10].apply(find_number_in_tag).tolist()
    # and add them to a list
    number_list = [int(i) for i in number_list if i is not None]
    if len(number_list) == 0:  # if no numbers were found
        df[10] = df[10].apply(lambda x: '0')  # add a zero instead of adding subcolumns
    else:  # if not
        max_number = max(number_list)  # find the highest number ID
        # and use the subfunction to add subcolumns to the df
        df[10] = df[10].apply(lambda x: subfunction_subcolumns(x, max_number))
    return df


def insts_to_conll(insts_dir, outdir):
    """
    Turns output files from the model into the right conll format

    :insts_dir: directory containing the predictions
    :outdir: directory where the conll-formatted files are added
    """
    dir_len = len(insts_dir)  # helps obtain the filename later
    for subdir, dirs, files in os.walk(insts_dir):
        if len(files) > 0:  # if in a directory that contains files
            file_df = pd.DataFrame()  # construct a dataframe
            chains_count = 0  # set a chain counter
            for sent_count in range(len(files)):
                # load in the pickle files
                df = joblib.load(f'{subdir}\\{sent_count}.pickle')
                # update the df and chain count 
                df, chains_count = add_biotags(df, chains_count)
                # and add everything to the general file df
                file_df = pd.concat([file_df, df])
            # construct a path to write to
            outfile = os.path.join(outdir, subdir[dir_len:])
            outfile += '.system'  # add a system file extension
            # add subcolumns
            file_df = add_subcolumns(file_df, sent_count)
            # slice off sentence indices
            file_df = file_df.iloc[:, :11]
            # and write the file
            file_df.to_csv(outfile, sep='\t', header=False, index=False)


insts_to_conll('.\\final_output (1)\\final_output\\', '.\\results\\')
