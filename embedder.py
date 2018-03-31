__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"


import pandas
import os
import numpy as np
from glove_loader import DIM

def save_embeddings(mode='texts'):
    dataset_name = 'cdcp_ACL17'
    dataset_version = 'new_2'
    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
    dataframe_path = os.path.join(dataset_path, 'pickles', dataset_version, 'total.pkl')
    df = pandas.read_pickle(dataframe_path)
    embeddings_path = os.path.join(dataset_path, 'embeddings', dataset_version)
    # load glove vocabulary and embeddings
    vocabulary_path = os.path.join(dataset_path, 'glove', 'glove.embeddings.npz')
    vocabulary_list = np.load(vocabulary_path)
    embed_list = vocabulary_list['embeds']
    word_list = vocabulary_list['vocab']

    vocabulary = {}
    for index in range(len(word_list)):
        vocabulary[word_list[index]] = embed_list[index]

    df_text = []
    if mode == 'texts':
        df_text = df[['text_ID', 'rawtext']].drop_duplicates()
    elif mode == 'propositions':
        df_text = df[['source_ID', 'source_proposition']].drop_duplicates()

    separators = ['(', ')', '[', ']', '...', '_', '--',
                  ';', ':',
                  '!!!', '???', '?!?', '!?!', '?!', '!?', '??', '!!',
                  '!', '?',
                  '/', '"', '\'\'', '%', '$', '*', '#', '+',
                  ',', '.',
                  '\'s', '\'ve', '\'ll', '\'re', '\'d',
                  '-', '\'']

    for index, (text_id, text) in df_text.iterrows():
        splits = text.split()
        tokens = ['']*len(splits)

        for i in range(len(splits)):
            word = splits[i]
            if word in vocabulary.keys():
                tokens[i] = word

        for separator in separators:
            i = 0
            while i < len(splits):
                if tokens[i] == '' and splits[i] != '':
                    index = 0
                    prev_index = 0
                    while index < len(splits[i]) and index >= 0:
                        word = splits[i]
                        index = word.find(separator, index)
                        if index >= 0:
                            prev_word = word[prev_index:index]
                            next_word = word[index+len(separator):]
                            if prev_word != '':
                                splits.insert(i, prev_word)
                                token = ''
                                if prev_word in vocabulary.keys():
                                    token = prev_word
                                tokens.insert(i, token)
                                i += 1

                            # adds the separator
                            splits[i] = separator
                            tokens[i] = separator

                            # avoids finding the same separator too many times
                            index += len(separator)

                            if next_word != '':
                                splits.insert(i+1, next_word)
                                token = ''
                                if next_word in vocabulary.keys():
                                    token = next_word
                                tokens.insert(i+1, token)
                i += 1



        embeddings = []
        for token in tokens:
            if token == '':
                print(text_id)
                print(text)
                print(tokens)
                print()
            else:
                embeddings.append(vocabulary[token])
        embeddings = np.array(embeddings, dtype=np.float32)

        if mode == 'texts':
            name = "%05d" % text_id + ".npz"
        elif mode == 'propositions':
            name = text_id

        document_path = os.path.join(embeddings_path, name)
        np.savez(document_path, embeddings)

        global MAX
        max = len(embeddings)
        if max > MAX:
            MAX = max

    print("Finished")


if __name__ == '__main__':
    global MAX
    MAX = 0
    save_embeddings('propositions')
    print(MAX)
    MAX = 0
    save_embeddings('texts')
    print(MAX)
