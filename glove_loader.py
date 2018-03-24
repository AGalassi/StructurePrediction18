import pandas
import os
import numpy as np

def vocabulary_creator(vocabulary_source_path, vocabulary_destination_path, dataframe_path):
    vocabulary = {}
    orphans = []
    vocabulary_destination_path = os.path.join(dataset_path, 'glove.txt')
    orphans_destination_path = os.path.join(dataset_path, 'orphans.txt')

    print("Loading Glove")

    f = open(vocabulary_source_path, 'r', encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        # embedding = np.array([float(val) for val in splitLine[1:]])
        # model[word] = embedding
        model[word] = line

    print("Glove loaded")

    df = pandas.read_pickle(dataframe_path)

    text_list = df['proposition_source'].drop_duplicates()

    for text in text_list:
        words = text.split()
        for word in words:
            if word in model.keys():
                vocabulary[word] = model[word]
            else:
                orphans.append(word)

    voc_file = open(vocabulary_destination_path,'w')
    voc_file.write(vocabulary)
    voc_file.close()
    orphans_file = open(orphans_destination_path, 'w')
    orphans_file.write(orphans)
    orphans_file.close()



dataset_name = 'cdcp_ACL17'
dataset_path = os.path.join(os.getcwd(),'Datasets', dataset_name)
pickles_path = os.path.join(os.path.join(dataset_path, 'pickles'))
dataframe_path = os.path.join(pickles_path, 'total.pkl')
vocabulary_source_path = os.path.join(os.getcwd(), 'glove.840B.300d.txt')

vocabulary_creator(vocabulary_source_path, dataset_path, dataframe_path)