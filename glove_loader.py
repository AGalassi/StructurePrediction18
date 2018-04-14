__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"


import pandas
import os
import numpy as np
import re

DIM = 300
SEPARATORS = ['(', ')', '[', ']', '...', '_', '--',
              ';', ':',
              '!!!', '???', '?!?', '!?!', '?!', '!?', '??', '!!',
              '!', '?',
              '/', '"', '%', '$', '*', '#', '+',
              ',', '.',
              "'s", "'ve", "'ll", "'re", "'d",
              '-', "'"]

STOPWORDS = ['.', ',', ':', ';']

def load_glove(vocabulary_source_path):

    print("Loading Glove")
    f = open(vocabulary_source_path, 'r', encoding="utf8")
    model = {}

    for line in f:
        splits = line.split(' ')
        n_splits = len(splits)
        word = ""
        n = 0
        while (n_splits - n) > DIM:
            word += " " + splits[n]
            n += 1
        word = word[1:]
        # embedding = np.array([float(val) for val in splitLine[1:]])
        # model[word] = embedding
        model[word] = line

    print("Glove loaded")
    return model



def vocabulary_creator(model, vocabulary_destination_path, dataframe_path):


    df = pandas.read_pickle(dataframe_path)

    propositions = df['source_proposition'].drop_duplicates()

    documents = []
    for proposition in propositions:
        proposition = proposition.replace("’", "'")
        proposition = proposition.replace("‘", "'")
        proposition = proposition.replace("“", '"')
        proposition = proposition.replace("”", '"')
        proposition = proposition.replace("''", '"')
        documents.append(proposition)


    if not os.path.exists(vocabulary_destination_path):
        os.makedirs(vocabulary_destination_path)
    orphans_path = os.path.join(vocabulary_destination_path, 'glove.orphans.txt')
    embeddings_path = os.path.join(vocabulary_destination_path, 'glove.embeddings.txt')
    npz_path = os.path.join(vocabulary_destination_path, 'glove.embeddings.npz')
    vocabulary_path = os.path.join(vocabulary_destination_path, 'glove.vocabulary.txt')
    logfile_path = os.path.join(vocabulary_destination_path, 'glove.log.txt')
    logfile = open(logfile_path, 'w')
    logfile.write('Sep\tVoc_size\tOrphans\n')

    print("Splitting")

    vocabulary, orphans = document_tokenizer_and_embedder(documents, model, logfile)


    logfile.close()
    orphans.remove('')


    # print vocabulary file
    vocabulary_file = open(vocabulary_path, 'w')
    for word in sorted(vocabulary.keys()):
        vocabulary_file.write(word)
        vocabulary_file.write('\n')
    vocabulary_file.close()

    # print orphans file
    orphans_file = open(orphans_path, 'w')
    for word in sorted(orphans):
        orphans_file.write(word)
        orphans_file.write("\n")
    orphans_file.close()

    print("handling orphans")

    # create random embeddings for orphans
    for word in sorted(orphans):
        embedding = np.random.rand(DIM) - 0.5
        line = word + " "
        for value in embedding:
            line += ("%.5g " % value) + " "
        line += '\n'
        vocabulary[word] = line

    print("Saving")

    # save embeddings
    embeddings = []
    vocabulary_list = []
    embeddings_file = open(embeddings_path, 'w')
    for word in sorted(vocabulary.keys()):
        line = vocabulary[word]
        embeddings_file.write(line)
        splits = line.split()
        embedding = splits[-DIM:]
        embedding = np.array(embedding, dtype=np.float32)
        embeddings.append(embedding)
        vocabulary_list.append(word)
    embeddings_file.close()

    print(vocabulary_list[0])

    np.savez(npz_path, vocab=vocabulary_list, embeds=embeddings)

    print('Finished')


def print_vocabulary_and_orphans(vocabulary, vocabulary_path, orphans, orphans_path):
    voc_file = open(vocabulary_path,'w')
    for word in sorted(vocabulary.keys()):
        voc_file.write(vocabulary[word])
    voc_file.close()
    orphans_file = open(orphans_path, 'w')
    for word in sorted(orphans):
        orphans_file.write(word)
        orphans_file.write("\n")
    orphans_file.close()


def document_tokenizer_and_embedder(documents, model,
                                    logfile=None, vocabulary={}, separators=None, not_vocab_separators=None):
    """
        Split the documents in tokens.
        The splitting is progressive using a series of separators,
        when a token match a key in model, it is inserted in the vocabulary.
        At the end of the process, the token that still do not match the model are returned as "orphans".

        Parameters
        ----------
        documents : an iterable object composed by str
            A list or a set of documents to be splitted.
        model : dict
            A dictionary with all the possible tokens as key
        logfile : file, optional
            File where to print the log of the tokenization process
        vocabulary : dict, optional
            The dictionary to be filled with the tokens found in the document splitting.
            If it is not provided, a empty dictionary is initialized.
        separators : list of str
            String to be used as splitting tokens.
            They will be inserted in the vocabulary if they are not in the next param
        not_vocab_separators : list of str
            Separators that will not be added to the vocabulary.

        Returns
        ----------
        orphans : list of str
            Token that do not match the model
        vocabulary : dict
            The keys are the tokens found during the splitting, the values come from the model
    """

    # punctuation and other special espressions
    if separators == None:
        separators = SEPARATORS
    # tried but not present in glove: '\'t', 'e-'

    orphans = set()
    for composed_word in documents:
        words = composed_word.split()
        #words = filter(None, re.split("[" + separator + "]+", composed_word))
        # remove stop symbols at the end of the tokens
        for word in words:
            if len(word) > 1 and word[-1] in STOPWORDS:
                word2 = word[:-1]
                if word2 in model.keys():
                    word = word2

            if word in model.keys():
                vocabulary[word] = model[word]
                # print("Found word: " + word)
            else:
                orphans.add(word)
                # print("Word not found: " + word)


    if not logfile == None:
        logfile.write("Tab, space, newline" + '\t' +
                      str(len(vocabulary.keys())) + '\t' +
                      str(len(orphans)) + '\n')

    for separator in separators:
        # print("Separator: " + separator)
        orphans, vocabulary = regular_split(orphans, vocabulary, model, separator)
        vocabulary[separator] = model[separator]
        # print("Orphans: " + str(len(orphans)))

        if not logfile == None:
            logfile.write(separator + '\t' +
                          str(len(vocabulary.keys())) + '\t' +
                          str(len(orphans)) + '\n')

    return vocabulary, orphans


def regular_split(old_orphans, vocabulary, model, separator):
    orphans = set()
    for composed_word in old_orphans:
        words = composed_word.split(separator)
        #words = filter(None, re.split("[" + separator + "]+", composed_word))
        for word in words:
            if word in model.keys():
                vocabulary[word] = model[word]
                # print("Found word: " + word)
            else:
                orphans.add(word)
                # print("Word not found: " + word)
    return orphans, vocabulary


if __name__ == '__main__':

    vocabulary_source_path = os.path.join(os.getcwd(), 'glove.840B.300d.txt')

    model = load_glove(vocabulary_source_path)

    m1 = model.copy()

    dataset_name = 'AAEC_v2'
    version = 'new_2'

    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
    pickles_path = os.path.join(os.path.join(dataset_path, 'pickles'))
    version_path = os.path.join(os.path.join(pickles_path, version))
    dataframe_path = os.path.join(version_path, 'total.pkl')
    glove_path = os.path.join(dataset_path, 'glove', version)

    vocabulary_creator(m1, glove_path, dataframe_path)


    dataset_name = 'cdcp_ACL17'
    version = 'new_3'

    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
    pickles_path = os.path.join(os.path.join(dataset_path, 'pickles'))
    version_path = os.path.join(os.path.join(pickles_path, version))
    dataframe_path = os.path.join(version_path, 'total.pkl')
    glove_path = os.path.join(dataset_path, 'glove', version)

    vocabulary_creator(model, glove_path, dataframe_path)
