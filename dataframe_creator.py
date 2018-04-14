__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"


import os
import pandas
import json
import random
import numpy as np


def split_propositions(text, propositions_offsets):
    propositions = []
    for offsets in propositions_offsets:
        propositions.append(text[offsets[0]:offsets[1]])
    return propositions

"""
def create_original_dataset_pickle(dataset_path, link_types, dataset_type='train'):
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
"""

def create_preprocessed_cdcp_pickle(dataset_path, dataset_version, link_types, dataset_type='train', validation=0, reflexive=True):
    data_path = os.path.join(dataset_path, dataset_version, dataset_type)

    normal_list = []
    validation_list = []

    prop_counter = {}
    rel_counter = {}
    val_prop_counter = {}
    val_rel_counter = {}

    for i in range (2000):
        file_name = "%05d" % (i)
        text_file_path = os.path.join(data_path, file_name + ".txt")
        if os.path.exists(text_file_path):

            split = dataset_type

            if validation > 0 and validation < 1:
                p = random.random()
                if p < validation:
                    split = 'validation'

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

            if (num_propositions <= 1):
                print('YEP!')


            for sourceID in range(num_propositions):

                type1 = data['prop_labels'][sourceID]

                for targetID in range(num_propositions):
                    if sourceID == targetID and not reflexive:
                        continue
                    relation_type = None
                    relation1to2 = False

                    # relation type
                    for link_type in link_types:
                        links = data[link_type]

                        for link in links:
                            # DEBUG
                            # if not link[0][0] == link[0][1]:
                                # raise Exception('MORE PROPOSITIONS IN THE SAME RELATION: document ' + file_name)

                            if link[0] == sourceID and link[1] == targetID:
                                if relation_type is not None and not relation_type == link_type:
                                    raise Exception('MORE RELATION FOR THE SAME PROPOSITIONS: document ' + file_name)
                                relation_type = link_type
                                relation1to2 = True

                            elif link[0] == targetID and link[1] == sourceID:
                                relation_type = "inv_" + link_type

                    # proposition type
                    type2 = data['prop_labels'][targetID]

                    dataframe_row = {'text_ID': i,
                                     'rawtext': raw_text,
                                     'source_proposition': propositions[sourceID],
                                     'source_ID': str(i) + "_" + str(sourceID),
                                     'target_proposition': propositions[targetID],
                                     'target_ID': str(i) + "_" + str(targetID),
                                     'source_type': type1,
                                     'target_type': type2,
                                     'relation_type': relation_type,
                                     'source_to_target': relation1to2,
                                     'set': split
                                     }

                    if split == 'validation':
                        validation_list.append(dataframe_row)

                        if relation_type not in val_rel_counter.keys():
                            val_rel_counter[relation_type] = 0
                        val_rel_counter[relation_type] += 1
                    else:
                        normal_list.append(dataframe_row)

                        if relation_type not in rel_counter.keys():
                            rel_counter[relation_type] = 0
                        rel_counter[relation_type] += 1


                if split == 'validation':
                    if type1 not in val_prop_counter.keys():
                        val_prop_counter[type1] = 0
                    val_prop_counter[type1] += 1
                else:
                    if type1 not in prop_counter.keys():
                        prop_counter[type1] = 0
                    prop_counter[type1] += 1

    pickles_path = os.path.join(dataset_path, 'pickles', dataset_version)
    if not os.path.exists(pickles_path):
        os.makedirs(pickles_path)

    if len(normal_list)>0:
        dataframe = pandas.DataFrame(normal_list)

        dataframe = dataframe[['text_ID',
                               'rawtext',
                               'source_proposition',
                               'source_ID',
                               'target_proposition',
                               'target_ID',
                               'source_type',
                               'target_type',
                               'relation_type',
                               'source_to_target',
                               'set']]

        dataframe_path = os.path.join(pickles_path, dataset_type + ".pkl")
        dataframe.to_pickle(dataframe_path)

    if len(validation_list) > 0:
        dataframe = pandas.DataFrame(validation_list)

        dataframe = dataframe[['text_ID',
                               'rawtext',
                               'source_proposition',
                               'source_ID',
                               'target_proposition',
                               'target_ID',
                               'source_type',
                               'target_type',
                               'relation_type',
                               'source_to_target',
                               'set']]

        dataframe_path = os.path.join(pickles_path, 'validation' + ".pkl")
        dataframe.to_pickle(dataframe_path)

    print("_______________")
    print(dataset_type)
    print(prop_counter)
    print(rel_counter)
    print("_______________")
    print("VALIDATION")
    print(val_prop_counter)
    print(val_rel_counter)
    print("_______________")



ukp_train_ids = [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53,
                 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 69, 70,
                 73, 74, 75, 76, 78, 79, 80, 81, 83, 84, 85, 87, 88, 89, 90,
                 92, 93, 94, 95, 96, 99, 100, 101, 102, 105, 106, 107, 109,
                 110, 111, 112, 113, 114, 115, 116, 118, 120, 121, 122, 123,
                 124, 125, 127, 128, 130, 131, 132, 133, 134, 135, 137, 138,
                 140, 141, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153,
                 155, 156, 157, 158, 159, 161, 162, 164, 165, 166, 167, 168,
                 170, 171, 173, 174, 175, 176, 177, 178, 179, 181, 183, 184,
                 185, 186, 188, 189, 190, 191, 194, 195, 196, 197, 198, 200,
                 201, 203, 205, 206, 207, 208, 209, 210, 213, 214, 215, 216,
                 217, 219, 222, 223, 224, 225, 226, 228, 230, 231, 232, 233,
                 235, 236, 237, 238, 239, 242, 244, 246, 247, 248, 249, 250,
                 251, 253, 254, 256, 257, 258, 260, 261, 262, 263, 264, 267,
                 268, 269, 270, 271, 272, 273, 274, 275, 276, 279, 280, 281,
                 282, 283, 284, 285, 286, 288, 290, 291, 292, 293, 294, 295,
                 296, 297, 298, 299, 300, 302, 303, 304, 305, 307, 308, 309,
                 311, 312, 313, 314, 315, 317, 318, 319, 320, 321, 323, 324,
                 325, 326, 327, 329, 330, 332, 333, 334, 336, 337, 338, 339,
                 340, 342, 343, 344, 345, 346, 347, 349, 350, 351, 353, 354,
                 356, 357, 358, 360, 361, 362, 363, 365, 366, 367, 368, 369,
                 370, 371, 372, 374, 375, 376, 377, 378, 379, 380, 381, 383,
                 384, 385, 387, 388, 389, 390, 391, 392, 394, 395, 396, 397,
                 399, 400, 401, 402]

ukp_test_ids = [4, 5, 6, 21, 42, 52, 61, 68, 71, 72, 77, 82, 86, 91, 97, 98,
                103, 104, 108, 117, 119, 126, 129, 136, 139, 142, 149, 154,
                160, 163, 169, 172, 180, 182, 187, 192, 193, 199, 202, 204,
                211, 212, 218, 220, 221, 227, 229, 234, 240, 241, 243, 245,
                252, 255, 259, 265, 266, 277, 278, 287, 289, 301, 306, 310,
                316, 322, 328, 331, 335, 341, 348, 352, 355, 359, 364, 373,
                382, 386, 393, 398]




def create_ukp_pickle(dataset_path, dataset_version, link_types, dataset_type='train', validation=0, reflexive=True):
    data_path = os.path.join(dataset_path, dataset_version)

    normal_list = []
    validation_list = []

    idlist = []

    if (dataset_type=='train'):
        idlist = ukp_train_ids
    elif (dataset_type=='test'):
        idlist = ukp_test_ids
    else:
        idlist = range(500)

    prop_counter = {}
    rel_counter = {}
    val_prop_counter = {}
    val_rel_counter = {}


    for i in idlist:
        file_name = "essay" + "%03d" % (i)
        text_file_path = os.path.join(data_path, file_name + ".txt")
        if os.path.exists(text_file_path):

            split = dataset_type

            if validation > 0 and validation < 1:
                p = random.random()
                if p < validation:
                    split = 'validation'

            text_file = open(text_file_path, 'r', encoding='utf-8')
            labels_file = open(os.path.join(data_path, file_name + ".ann"), 'r')

            labels_line = []

            raw_text = text_file.read()
            for splits in labels_file.read().split('\n'):
                labels_line.append(splits)

            text_file.close()
            labels_file.close()

            # elaborate the offsets of the paragraphs
            paragraphs_offsets = []
            start = 0
            while start < len(raw_text):
                try:
                    end = raw_text.index("\n", start)
                except ValueError:
                    end = len(raw_text)
                if end != start:
                    paragraphs_offsets.append([start, end])
                start = end + 1

            data = {'prop_labels': {},
                    'prop_offsets': {},
                    'start_offsets': {},
                    'T_ids': [],
                    'propositions': {}}

            for link_type in link_types:
                data[link_type] = []

            paragraphs = split_propositions(raw_text, paragraphs_offsets)

            for line in labels_line:
                splits = line.split(maxsplit=4)
                if len(splits) <= 0:
                    continue
                if splits[0][0] == 'T':
                    T_id = int(splits[0][1:])-1
                    data['T_ids'].append(T_id)
                    data['prop_labels'][T_id] = splits[1]
                    data['prop_offsets'][T_id] = [int(splits[2]), int(splits[3])]
                    data['start_offsets'][int(splits[2])] = T_id
                    data['propositions'][T_id] = splits[4].split('\n')
                elif splits[0][0] == 'R':
                    source = int(splits[2][6:]) - 1
                    target = int(splits[3][6:]) - 1
                    data[splits[1]].append([source, target])

            # new order given by the start offsets
            new_order = {}
            new_id = 0
            # find the match between the starting offsets and set the new id
            for new_off in sorted(data['start_offsets'].keys()):
                for old_id in data['T_ids']:
                    old_off = data['prop_offsets'][old_id][0]
                    if new_off == old_off:
                        new_order[old_id] = new_id
                        new_id += 1
                        break

            new_data = {'prop_labels': [-1]*len(data['prop_labels']),
                        'prop_offsets': [-1]*len(data['prop_labels']),
                        'propositions': [-1]*len(data['prop_labels']),}

            for link_type in link_types:
                new_data[link_type] = []

            for link_type in link_types:
                for link in data[link_type]:
                    old_source = link[0]
                    old_target = link[1]
                    new_source = new_order[old_source]
                    new_target = new_order[old_target]
                    new_data[link_type].append([new_source, new_target])

            for old_id in data['T_ids']:
                new_id = new_order[old_id]
                new_data['prop_labels'][new_id] = data['prop_labels'][old_id]
                new_data['prop_offsets'][new_id] = data['prop_offsets'][old_id]
                new_data['propositions'][new_id] = data['propositions'][old_id]

            data = new_data

            propositions = data['propositions']

            num_propositions = len(propositions)

            assert (num_propositions >= 1)

            for sourceID in range(num_propositions):

                source_start = data['prop_offsets'][sourceID][0]
                p_offsets = (-1, -1)
                par = -1
                # find the paragraph
                for paragraph in range(len(paragraphs)):
                    p_start = paragraphs_offsets[paragraph][0]
                    p_end = paragraphs_offsets[paragraph][1]
                    if p_end >= source_start >= p_start:
                        p_offsets = (p_start, p_end)
                        par = paragraph

                assert par != -1
                type1 = data['prop_labels'][sourceID]

                for targetID in range(num_propositions):
                    # proposition type
                    type2 = data['prop_labels'][targetID]

                    target_start = data['prop_offsets'][targetID][0]

                    if sourceID == targetID and not reflexive:
                        continue

                    # relations in different paragraphs are not allowed
                    if target_start < p_offsets[0] or target_start > p_offsets[1]:
                        continue

                    relation_type = None
                    relation1to2 = False

                    # relation type
                    for link_type in link_types:
                        links = data[link_type]

                        for link in links:
                            # DEBUG
                            # if not link[0][0] == link[0][1]:
                            # raise Exception('MORE PROPOSITIONS IN THE SAME RELATION: document ' + file_name)

                            if link[0] == sourceID and link[1] == targetID:
                                if relation_type is not None and not relation_type == link_type:
                                    raise Exception('MORE RELATION FOR THE SAME PROPOSITIONS: document ' + file_name)
                                relation_type = link_type
                                relation1to2 = True

                            elif link[0] == targetID and link[1] == sourceID:
                                relation_type = "inv_" + link_type

                    dataframe_row = {'text_ID': str(i) + "_" + str(par),
                                     'rawtext': paragraphs[par],
                                     'source_proposition': propositions[sourceID],
                                     'source_ID': str(i) + "_" + str(par) + "_" + str(sourceID),
                                     'target_proposition': propositions[targetID],
                                     'target_ID': str(i) + "_" + str(par) + "_" + str(targetID),
                                     'source_type': type1,
                                     'target_type': type2,
                                     'relation_type': relation_type,
                                     'source_to_target': relation1to2,
                                     'set': split
                                     }

                    if split == 'validation':
                        validation_list.append(dataframe_row)

                        if relation_type not in val_rel_counter.keys():
                            val_rel_counter[relation_type] = 0
                        val_rel_counter[relation_type] += 1
                    else:
                        normal_list.append(dataframe_row)

                        if relation_type not in rel_counter.keys():
                            rel_counter[relation_type] = 0
                        rel_counter[relation_type] += 1

                if split == 'validation':
                    if type1 not in val_prop_counter.keys():
                        val_prop_counter[type1] = 0
                    val_prop_counter[type1] += 1
                else:
                    if type1 not in prop_counter.keys():
                        prop_counter[type1] = 0
                    prop_counter[type1] += 1

    pickles_path = os.path.join(dataset_path, 'pickles', dataset_version)
    if not os.path.exists(pickles_path):
        os.makedirs(pickles_path)

    if len(normal_list) > 0:
        dataframe = pandas.DataFrame(normal_list)

        dataframe = dataframe[['text_ID',
                               'rawtext',
                               'source_proposition',
                               'source_ID',
                               'target_proposition',
                               'target_ID',
                               'source_type',
                               'target_type',
                               'relation_type',
                               'source_to_target',
                               'set']]

        dataframe_path = os.path.join(pickles_path, dataset_type + ".pkl")

        dataframe.to_pickle(dataframe_path)

    if len(validation_list) > 0:
        dataframe = pandas.DataFrame(validation_list)

        dataframe = dataframe[['text_ID',
                               'rawtext',
                               'source_proposition',
                               'source_ID',
                               'target_proposition',
                               'target_ID',
                               'source_type',
                               'target_type',
                               'relation_type',
                               'source_to_target',
                               'set']]

        dataframe_path = os.path.join(pickles_path, 'validation' + ".pkl")
        dataframe.to_pickle(dataframe_path)

    print("_______________")
    print(dataset_type)
    print(prop_counter)
    print(rel_counter)
    print("_______________")
    print("VALIDATION")
    print(val_prop_counter)
    print(val_rel_counter)
    print("_______________")


def print_dataframe_details(dataframe_path):
    df = pandas.read_pickle(dataframe_path)

    print()
    print('total relations')
    print(len(df))
    print()
    column = 'source_to_target'
    print(df[column].value_counts())
    print()
    column = 'relation_type'
    print(df[column].value_counts())

    print()
    column = 'text_ID'
    print(column)
    print(len(df[column].drop_duplicates()))

    print()
    column = 'source_ID'
    print(column)
    print(len(df[column].drop_duplicates()))




def create_total_dataframe(pickles_path):
    frames = []
    dataframe_path = os.path.join(pickles_path, 'train.pkl')
    df1 = pandas.read_pickle(dataframe_path)
    frames.append(df1)
    dataframe_path = os.path.join(pickles_path, 'test.pkl')
    df2 = pandas.read_pickle(dataframe_path)
    frames.append(df2)
    dataframe_path = os.path.join(pickles_path, 'validation.pkl')
    if os.path.exists(dataframe_path):
        df3 = pandas.read_pickle(dataframe_path)
        frames.append(df3)

    dataframe = pandas.concat(frames).sort_values('text_ID')

    dataframe_path = os.path.join(pickles_path, 'total.pkl')
    dataframe.to_pickle(dataframe_path)


if __name__ == '__main__':

    # """

    dataset_type = 'train'
    # dataset_name = 'cdcp_ACL17'
    # dataset_version = 'preprocessed+tran_data'
    dataset_name = 'AAEC_v2'
    dataset_version = 'original_data'
    create_function = None


    link_types = []


    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
    if dataset_name == 'cdcp_ACL17':
        link_types = ['evidences', 'reasons']
        create_function = create_preprocessed_cdcp_pickle
    elif dataset_name == 'AAEC_v2':
        link_types = ['supports', 'attacks']
        create_function = create_ukp_pickle

    create_function(dataset_path, dataset_version, link_types, dataset_type, validation=0.1, reflexive=False)
    dataset_type = 'test'
    create_function(dataset_path, dataset_version, link_types, dataset_type, reflexive=False)

    pickles_path = os.path.join(dataset_path, 'pickles', dataset_version)
    create_total_dataframe(pickles_path)


    print("----------------------\n\n")
    dataframe_path = os.path.join(pickles_path, 'validation.pkl')
    if os.path.exists(dataframe_path):
        print("VALIDATION")
        print_dataframe_details(dataframe_path)
    print("----------------------\n\n")
    print("TRAIN")
    dataframe_path = os.path.join(pickles_path, 'train.pkl')
    print_dataframe_details(dataframe_path)
    print("----------------------\n\n")
    print("TEST")
    dataframe_path = os.path.join(pickles_path, 'test.pkl')
    print_dataframe_details(dataframe_path)





    """
    # link_types = ['evidences', 'reasons']
    # dataset_name = 'cdcp_ACL17'
    # dataset_version = 'new_3'
    # i = 1

    dataset_name = 'AAEC_v2'
    dataset_version = 'new_2'
    link_types = ['supports', 'attacks']
    i = 2

    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
    split = 'total'
    dataframe_path = os.path.join(dataset_path, 'pickles', dataset_version, split + '.pkl')
    df = pandas.read_pickle(dataframe_path)

    diff_l = {}
    diff_nl = {}
    highest = 0

    for index, row in df.iterrows():
        s_index = int(row['source_ID'].split('_')[i])
        t_index = int(row['target_ID'].split('_')[i])

        difference = abs(s_index-t_index)

        if highest<difference:
            highest = difference

        if row['source_to_target']:
            voc = diff_l
        else:
            voc = diff_nl

        if difference in voc.keys():
            voc[difference] += 1
        else:
            voc[difference] = 1

    for key in range(1, highest+1):
        if key not in diff_nl.keys():
            diff_nl[key] = 0
        if key not in diff_l.keys():
            diff_l[key] = 0

        print(str(key) + "\t" + str(diff_nl[key]) + '\t' + str(diff_l[key]))

    """

