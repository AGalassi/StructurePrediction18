__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"


import pandas
import os
import numpy as np
import pickle
from glove_loader import DIM, SEPARATORS, STOPWORDS

def save_embeddings(dataset_name='cdcp_ACL17', dataset_version='new_2', mode='texts', type='embeddings'):
    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
    dataframe_path = os.path.join(dataset_path, 'pickles', dataset_version, 'total.pkl')
    df = pandas.read_pickle(dataframe_path)
    embeddings_path = os.path.join(dataset_path, type, dataset_version)
    # load glove vocabulary and embeddings
    vocabulary_path = os.path.join(dataset_path, 'glove', dataset_version, 'glove.embeddings.npz')
    vocabulary_list = np.load(vocabulary_path)
    embed_list = vocabulary_list['embeds']
    word_list = vocabulary_list['vocab']

    vocabulary = {}
    for index in range(len(word_list)):
        if type == 'embeddings':
            vocabulary[word_list[index]] = embed_list[index]
        elif type == 'bow':
            # the 0 index must be left empty for padding
            vocabulary[word_list[index]] = index + 1

    df_text = []
    if mode == 'texts':
        df_text = df[['text_ID', 'rawtext']].drop_duplicates()
    elif mode == 'propositions':
        df_text = df[['source_ID', 'source_proposition']].drop_duplicates()

    separators = SEPARATORS

    for index, (text_id, text) in df_text.iterrows():

        text = text.replace("’", "'")
        text = text.replace("‘", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')
        text = text.replace("''", '"')

        splits = text.split()
        tokens = ['']*len(splits)

        # initial split with common separators
        i = 0
        while i < len(splits):
            word = splits[i]

            # remove possible stop symbols in the end of the token
            if len(word) > 1 and word[-1] in STOPWORDS and word[:-1] in vocabulary.keys():
                symbol = word[-1]
                word = word[:-1]
                splits.insert(i + 1, symbol)
                tokens.insert(i + 1, symbol)
                splits[i] = word
                tokens[i] = word
                tokens[i] = word

            elif word in vocabulary.keys():
                tokens[i] = word

            i += 1

        for separator in separators:
            i = 0
            # iterate on the whole list of split, creating new splits with the separator
            while i < len(splits):
                # the word is not empty and is not recognized as a token
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
                                tokens.insert(i+1, token)
                i += 1

            # recognize tokens
            i = 0
            while i < len(splits):
                word = splits[i]
                # remove possible stop symbols in the end of the token
                if len(word) > 1 and word[-1] in STOPWORDS and word[:-1] in vocabulary.keys():
                    symbol = word[-1]
                    word = word[:-1]
                    splits.insert(i + 1, symbol)
                    tokens.insert(i + 1, symbol)
                    splits[i] = word
                    tokens[i] = word
                    tokens[i] = word

                elif word in vocabulary.keys():
                    tokens[i] = word

                i += 1



        embeddings = []
        for token in tokens:
            if token == '':
                print("TOKEN NOT RECOGNIZED!")
                print(text_id)
                print(text)
                print(tokens)
                print()
            else:
                embeddings.append(vocabulary[token])

        array_type = np.float32
        if type == 'bow':
            array_type = int

        embeddings = np.array(embeddings, dtype=array_type)

        if mode == 'texts':
            name = str(text_id) + ".npz"
        elif mode == 'propositions':
            name = str(text_id) + ".npz"

        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)

        document_path = os.path.join(embeddings_path, name)
        np.savez(document_path, embeddings)


        global MAX
        max = len(embeddings)
        if max > MAX:
            MAX = max

    print("Finished")


if __name__ == '__main__':
    global MAX
    # MAX = 0
    # save_embeddings('AAEC_v2', 'new_2', 'propositions', 'bow')
    # print(MAX)

    # MAX = 0
    # save_embeddings('cdcp_ACL17', 'new_3', 'propositions', 'bow')
    # print(MAX)

    MAX = 0
    save_embeddings('DrInventor', 'new_0', 'propositions', 'bow')
    print(MAX)
