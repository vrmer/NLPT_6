
def generate_attribution_column(df,pred):
    '''
    Given the input data frame and a dictionary of predictions (in which article is the first level,
    sentences the second level, tokens the third level, and tags the lowest level values), generate a list of attribution tags.
    :param df: pandas data frame resulting from reading the article file as with pd.read_csv().
    :param pred: the dictionary with article, sentences, tokens and respective tags as output by baseline system.
    :return: a list of attributation tags, where each tag corresponds to a token classification by baseline in conll format e.g. ___I-CONTENT. Each underscore represents an AR in the article.
    '''

    attributions = []  # the column with attributions in output file
    for article in df.article.unique(): # this is not necessary anymore, as each article is read in as a separate data frame
        # find how many cues are there per article to add same number of subcolumns to attribution column
        n_cues = 0
        for sent, tokens in pred[article].items():
             if "B-CUE" in tokens.values():
                n_cues += 1
        # add tags to the subcolumn of its AR
        df_article = df.loc[df.article == article]
        subcol_idx = 0
        for sent in df_article.sent_n.unique():
            # tags = pred[article][sent].values()
            sorted_items = sorted(pred[article][sent].items()) # sort sentence dict keys by token index in ascending order to match their order in the data frame
            tags = [item[1] for item in sorted_items]
            if any(label in tags for label in ['B-SOURCE', 'B-CONTENT', 'B-CUE', 'I-SOURCE', 'I-CONTENT', 'I-CUE']):
                for tag in tags:
                    att = ['_' for cue in range(n_cues)]  # add as many subcolumns as the number of cues in the article
                    att[subcol_idx] = tag  # add the token tag to the subcolumn at position subcol_idx
                    attributions.append(' '.join(att))  # join subcolumns into a string separated with whitespaces
                subcol_idx += 1  # increase subcolumn position for the next AR
            else:  # if the sentence has no ARs, just add attribution with underscores
                for tag in tags:
                    att = ['_' for cue in range(n_cues)]
                    attributions.append(' '.join(att))

    return attributions

def write_out_output_file (df, output_filepath):
    '''
    Given a data frame with output in attribution column, add blank line between sentences and write out as tsv file.
    :param df: pandas dataframe with output in attribution column
    :param output_filepath: the filepath to which the output file should be written
    '''
    df_rows = df.values.tolist() # transform rows of data frame into lists
    with open(output_filepath, 'w') as tsvfile:
        for row in df_rows:
            if row[3] == 1 and row != df_rows[0]: # if token index is 1, a new sentence begins, so add new line
                tsvfile.write('\n')  # add blank lines between sentences
            row = [str(cell) for cell in row]
            line = '\t'.join(row) + '\n' # join row cells into a string separated by tab
            tsvfile.write(line) # write out row