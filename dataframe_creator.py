__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "MIT"
__version__ = "1.1.1"
__email__ = "a.galassi@unibo.it"


import os
import pandas
import json
import numpy as np
import matplotlib.pyplot as plt


def split_propositions(text, propositions_offsets):
    propositions = []
    for offsets in propositions_offsets:
        propositions.append(text[offsets[0]:offsets[1]])
    return propositions


def create_dataset_pickle(dataset_path, link_types, dataset_type='train'):
    data_path = os.path.join(dataset_path, 'original_data', dataset_type)

    data_list = []

    for i in range (2000):
        file_name = "%05d" % (i)
        text_file_path = os.path.join(data_path, file_name + ".txt")
        if os.path.exists(text_file_path):

            text_file = open(text_file_path, 'r')
            labels_file = open(os.path.join(data_path, file_name + ".ann.json"), 'r')
            data = json.load(labels_file)
            raw_text = text_file.read()
            text_file.close()
            labels_file.close()

            propositions = split_propositions(raw_text, data['prop_offsets'])

            if len(data['url'])>0:
                print('URL! ' + str(i))

            num_propositions = len(propositions)

            for sen1 in range(num_propositions):
                for sen2 in range (num_propositions):
                    relation_type = None
                    relation1to2 = False

                    # relation type
                    for link_type in link_types:
                        links = data[link_type]

                        for link in links:
                            # DEBUG
                            # if not link[0][0] == link[0][1]:
                                # raise Exception('MORE PROPOSITIONS IN THE SAME RELATION: document ' + file_name)

                            source_range = range(link[0][0], link[0][1]+1)

                            if  sen1 in source_range and link[1] == sen2:
                                if relation_type is not None and not relation_type == link_type:
                                    raise Exception('MORE RELATION FOR THE SAME PROPOSITIONS: document ' + file_name)
                                relation_type = link_type
                                relation1to2 = True

                            elif sen2 in source_range and link[1] == sen1:
                                relation_type = "inv_" + link_type

                    # proposition type
                    type1 = data['prop_labels'][sen1]
                    type2 = data['prop_labels'][sen2]

                    dataframe_row = {'textID' : i,
                                     'rawtext': raw_text,
                                     'source_proposition':propositions[sen1],
                                     'target_proposition':propositions[sen2],
                                     'source_type':type1,
                                     'target_type':type2,
                                     'relation_type':relation_type,
                                     'source_to_target':relation1to2,
                                     'set':dataset_type
                    }

                    data_list.append(dataframe_row)


    dataframe = pandas.DataFrame(data_list)

    dataframe = dataframe[['textID',
                           'rawtext',
                           'source_proposition',
                           'target_proposition',
                           'source_type',
                           'target_type',
                           'relation_type',
                           'source_to_target',
                           'set']]

    pickles_path = os.path.join(os.path.join(dataset_path, 'pickles'))
    dataframe_path = os.path.join(pickles_path, dataset_type + ".pkl")

    if not os.path.exists(pickles_path):
        os.makedirs(pickles_path)
    dataframe.to_pickle(dataframe_path)


def print_dataframe_details(dataframe_path):
    df = pandas.read_pickle(dataframe_path)

    print()
    print('total')
    print(len(df))
    print()
    column = 'source_to_target'
    print(df[column].value_counts())
    print()
    column = 'relation_type'
    print(df[column].value_counts())

    print()
    column = 'textID'
    print(column)
    print(len(df[column].drop_duplicates()))


    print()
    column = 'source_proposition'
    af = df[['textID', 'source_proposition', 'source_type']].drop_duplicates()
    print(column)
    print(len(af[column]))


    af = df[['textID', 'source_proposition', 'source_type']].drop_duplicates()
    print()
    column = 'source_type'
    print(df[column].value_counts())


def create_total_dataframe(dataset_path):
    pickles_path = os.path.join(os.path.join(dataset_path, 'pickles'))
    dataframe_path = os.path.join(pickles_path, 'train.pkl')
    df1 = pandas.read_pickle(dataframe_path)
    dataframe_path = os.path.join(pickles_path, 'test.pkl')
    df2 = pandas.read_pickle(dataframe_path)

    frames = [df1, df2]
    dataframe = pandas.concat(frames).sort_values('textID')

    dataframe_path = os.path.join(pickles_path, 'total.pkl')
    dataframe.to_pickle(dataframe_path)


dataset_type = 'test'
link_types = ['evidences', 'reasons']
dataset_name = 'cdcp_ACL17'
dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
create_dataset_pickle(dataset_path, link_types, dataset_type)

