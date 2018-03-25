import pandas
import os
import numpy as np
import re

def vocabulary_creator(vocabulary_source_path, vocabulary_destination_path, dataframe_path):
    vocabulary = {}

    # punctuation and other special espressions
    separators = [' ', '(', ')', '[', ']', '...', '_', '--',
                  ';', ':', '!', '?',
                  '/', '"', '%', '$', '*', '#', '+',
                  ',', '.',
                  '\'s', '\'ve', '\'ll', '\'re', '\'d',
                  '-', '\'']
    # not present in glove: '\'t', 'e-'

    print("Loading Glove")

    f = open(vocabulary_source_path, 'r', encoding="utf8")
    model = {}
    for line in f:
        splits = line.split(' ')
        n_splits = len(splits)
        word = ""
        n = 0
        while (n_splits - n) > 300:
            word += " " + splits[n]
            n += 1
        word = word[1:]
        # embedding = np.array([float(val) for val in splitLine[1:]])
        # model[word] = embedding
        model[word] = line

    print("Glove loaded")

    df = pandas.read_pickle(dataframe_path)

    orphans = df['source_proposition'].drop_duplicates()

    if not os.path.exists(vocabulary_destination_path):
        os.makedirs(vocabulary_destination_path)
    orphans_path = os.path.join(vocabulary_destination_path, 'glove.orphans.txt')
    vocabulary_path = os.path.join(vocabulary_destination_path, 'glove.vocabulary.txt')
    logfile_path = os.path.join(vocabulary_destination_path, 'glove.log.txt')
    logfile = open(logfile_path, 'w')
    logfile.write('Sep\tVoc_size\tOrphans\n')

    for separator in separators:
        print("Separator: " + separator)
        orphans, vocabulary = regular_split(orphans, vocabulary, model, separator)
        if separator is not ' ':
            vocabulary[separator] = model[separator]
        print("Orphans: " + str(len(orphans)))

        logfile.write(separator + '\t' +
                      str(len(vocabulary.keys())) + '\t' +
                      str(len(orphans)) + '\n')

    logfile.close()

    voc_file = open(vocabulary_path, 'w')
    for word in sorted(vocabulary.keys()):
        voc_file.write(vocabulary[word])
    voc_file.close()

    orphans_file = open(orphans_path, 'w')
    for word in sorted(orphans):
        orphans_file.write(word)
        orphans_file.write("\n")
    orphans_file.close()

    string = ""
    for separator in separators:
        string += separator
    print(string)


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

dataset_name = 'cdcp_ACL17'
dataset_path = os.path.join(os.getcwd(),'Datasets', dataset_name)
pickles_path = os.path.join(os.path.join(dataset_path, 'pickles'))
dataframe_path = os.path.join(pickles_path, 'total.pkl')
vocabulary_source_path = os.path.join(os.getcwd(), 'glove.840B.300d.txt')
glove_path = os.path.join(dataset_path, 'glove')

vocabulary_creator(vocabulary_source_path, glove_path, dataframe_path)