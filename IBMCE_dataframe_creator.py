__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.1.0"
__email__ = "a.galassi@unibo.it"

import os
import pandas
import json
import random
import sys
import ast
import numpy as np


def split_propositions(text, propositions_offsets):
    propositions = []
    for offsets in propositions_offsets:
        propositions.append(text[offsets[0]:offsets[1]])
    return propositions


def create_IBM_pickle(dataset_path, dataset_version, link_types, test=0, validation=0, reflexive=False, distance=100):
    data_path = os.path.join(dataset_path, dataset_version)

    fake_sentences = ["[\t___URL___\t]\t[\t___URL___\t]\t[\t___URL___\t]\t",
                      "[\t___URL___\t]\t[\t___URL___\t]\t[\t___URL___\t]",
                      "\t[\t___URL___\t]\t[\t___URL___\t]\t[\t___URL___\t]\t",
                      "[\t___URL___\t]\t[\t___URL___\t]\t",
                      "[\t___URL___\t]\t[\t___URL___\t]",
                      "\t[\t___URL___\t]\t[\t___URL___\t]\t",
                      "[\t___URL___\t]\t", "[\t___URL___\t]", "\t[\t___URL___\t]\t",
                      "."]

    normal_list = []
    validation_list = []
    test_list = []

    idlist = range(6000)

    prop_counter = {}
    rel_counter = {}
    val_prop_counter = {}
    val_rel_counter = {}
    test_prop_counter = {}
    test_rel_counter = {}

    missing_claims = 0
    missing_evidences = 0
    tot_evidence = 0
    tot_claim = 0

    for i in idlist:
        text_file_path = os.path.join(data_path, "parsed", str(i) + "_parsed.txt")
        label_file_path = os.path.join(data_path, "labels", str(i) + "_labels.txt")
        if os.path.exists(label_file_path):

            data_split = "train"

            p = random.random()

            if validation > 0 and validation < 1 and test > 0 and test < 1:
                if p < validation:
                    data_split = 'validation'
                elif p < validation + test:
                    data_split = 'test'

            text_file = open(text_file_path, 'r', encoding='utf-8')
            text = text_file.read()
            text_lines = text.split('\n')

            # filter the lines: empty lines and not real sentences are not considered
            num_lines = len(text_lines)
            num = 0
            while num < num_lines:
                line = text_lines[num]
                if len(line) <= 0 or line in fake_sentences:
                    text_lines.remove(line)
                    num_lines = len(text_lines)
                    # print("REMOVED!")
                else:
                    num += 1

            label_file = open(label_file_path, 'r', encoding='utf-8')
            label_lines = label_file.read().split('\n')

            text_file.close()
            label_file.close()

            claims = set()
            evidences = set()
            sentences_claims = {}
            sentences_evidences = {}

            # for each document, select the claims and the evidences of the documents
            for line in label_lines:
                if len(line) <= 0:
                    continue
                splits = line.split("\t||||\t")
                claim = splits[1]
                evidence = splits[2]
                claims.add(claim)
                evidences.add(evidence)

            # for each document sentence, find out the list of claims and evidences it is part of
            c = 0
            for line in text_lines:
                c += 1
                if len(line) <= 0:
                    print("ERROR: EMPTY LINE!")
                    continue
                sentences_claims[line] = set()
                sentences_evidences[line] = set()
                for claim in claims:
                    claim_splits = claim.split("\t\t")
                    for split in claim_splits:
                        if split.lower() in line.lower():
                            sentences_claims[line].add(claim)
                            continue
                for evidence in evidences:

                    evidence_splits = evidence.split("\t\t")
                    for split in evidence_splits:
                        if ((split.lower() in line.lower()) and
                                (split not in fake_sentences) and
                                (len(split.split()) > 3)):
                            sentences_evidences[line].add(evidence)
                            continue

            tot_evidence += len(evidences)
            tot_claim += len(claims)

            # check if all the claims and evidences have been found
            for line in sentences_evidences.keys():
                for claim in sentences_claims[line]:
                    if claim in claims:
                        claims.remove(claim)
                for evidence in sentences_evidences[line]:
                    if evidence in evidences:
                        evidences.remove(evidence)

            # DEBUG: check the missing C/E
            """
            if len(claims) > 0:
                print("C")
                print(i)
                for claim in claims:
                    print(claim)
                print()

            if len(evidences) > 0:
                print("E")
                print(i)
                for evidence in evidences:
                    print(evidence)
                print()
            """

            missing_claims += len(claims)
            missing_evidences += len(evidences)

            # create couples
            for id_p1 in range(len(text_lines)):
                p1 = text_lines[id_p1]
                if len(p1) <= 0:
                    continue
                for id_p2 in range(len(text_lines)):
                    p2 = text_lines[id_p2]
                    if len(p2) <= 0:
                        continue
                    if id_p1 == id_p2 and not reflexive:
                        continue

                    prop_dist = id_p1 - id_p2
                    if abs(prop_dist) > distance > 0:
                        continue

                    # list of claims and evidences in the 2 sentences
                    c1 = sentences_claims[p1]
                    c2 = sentences_claims[p2]
                    e1 = sentences_evidences[p1]
                    e2 = sentences_evidences[p2]

                    if len(c1) > 0:
                        if len(e1) > 0:
                            type1 = "CE"
                        else:
                            type1 = "C"
                    elif len(e1) > 0:
                        type1 = "E"
                    else:
                        type1 = "N"

                    if len(c2) > 0:
                        if len(e2) > 0:
                            type2 = "CE"
                        else:
                            type2 = "C"
                    elif len(e2) > 0:
                        type2 = "E"
                    else:
                        type2 = "N"

                    relation1to2 = False
                    relation2to1 = False
                    relation_type = None

                    # since 2 propositions can have many relationship between them at the same time
                    study_rel = False
                    anecdotal_rel = False
                    expert_rel = False

                    for line in label_lines:
                        if len(line) <= 0:
                            continue
                        splits = line.split("\t||||\t")
                        claim = splits[1]
                        evidence = splits[2]
                        relationships = splits[3]

                        for claim1 in c1:
                            for evidence2 in e2:
                                if claim1 == claim and evidence2 == evidence:
                                    relation2to1 = True

                        for claim2 in c2:
                            for evidence1 in e1:
                                if claim2 == claim and evidence1 == evidence:
                                    relation1to2 = True

                        if relation1to2:
                            if "STUDY" in relationships:
                                study_rel = True
                            if "ANECDOTAL" in relationships:
                                anecdotal_rel = True
                            if "EXPERT" in relationships:
                                expert_rel = True

                    if relation1to2:
                        relation_type = ""
                        if study_rel:
                            relation_type += "S"
                        if anecdotal_rel:
                            relation_type += "A"
                        if expert_rel:
                            relation_type += "E"
                    elif relation2to1:
                        relation_type = "inverse"
                    else:
                        relation_type = "None"

                    dataframe_row = {'text_ID': str(i),
                                     'rawtext': text,
                                     'source_proposition': text_lines[id_p1],
                                     'source_ID': str(i) + "_" + str(id_p1),
                                     'target_proposition': text_lines[id_p2],
                                     'target_ID': str(i) + "_" + str(id_p2),
                                     'source_type': type1,
                                     'target_type': type2,
                                     'relation_type': relation_type,
                                     'source_to_target': relation1to2,
                                     'distance': prop_dist,
                                     'set': data_split
                                     }

                    if data_split == 'validation':
                        validation_list.append(dataframe_row)
                    elif data_split == 'test':
                        test_list.append(dataframe_row)
                    else:
                        normal_list.append(dataframe_row)

    if distance > 0:
        pickles_path = os.path.join(dataset_path, 'pickles', dataset_version + "_d" + str(distance))
    else:
        pickles_path = os.path.join(dataset_path, 'pickles', dataset_version)
    if not os.path.exists(pickles_path):
        os.makedirs(pickles_path)

    dataframes = []

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
                               'distance',
                               'set']]
        dataframes.append(dataframe)

        dataframe_path = os.path.join(pickles_path, "train.pkl")
        dataframe.to_pickle(dataframe_path)

    if len(test_list) > 0:
        dataframe = pandas.DataFrame(test_list)

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
                               'distance',
                               'set']]
        dataframes.append(dataframe)

        dataframe_path = os.path.join(pickles_path, 'test' + ".pkl")
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
                               'distance',
                               'set']]
        dataframes.append(dataframe)

        dataframe_path = os.path.join(pickles_path, 'validation' + ".pkl")
        dataframe.to_pickle(dataframe_path)

    dataframe = pandas.concat(dataframes)
    dataframe_path = os.path.join(pickles_path, 'complete' + ".pkl")
    dataframe.to_pickle(dataframe_path)

    print()
    print()
    print(missing_claims)
    print(tot_claim)
    print(missing_evidences)
    print(tot_evidence)


def find_IBM_claim_article(dataset_path):
    """
    Create a single file with all the information about the claims and the evidences
    :param dataset_path:
    :return:
    """
    dataset_path = os.path.join(dataset_path, 'original_data')
    article_path = os.path.join(dataset_path, 'articles')
    claims_path = os.path.join(dataset_path, 'claims.txt')
    topics_path = os.path.join(dataset_path, 'articles.txt')
    claims_path_new = os.path.join(dataset_path, 'claims_article.txt')
    evidence_path = os.path.join(dataset_path, 'evidence_corrected.txt')
    all_new = os.path.join(dataset_path, 'all.txt')

    claims_file = open(claims_path, 'r', encoding='utf-8')
    topics_file = open(topics_path, 'r', encoding='utf-8')
    evidence_file = open(evidence_path, 'r', encoding='utf-8')
    claims_file_new = open(claims_path_new, 'w', encoding='utf-8')
    all_file_new = open(all_new, 'w', encoding='utf-8')

    claims_text = claims_file.read()
    claims_list = claims_text.split('\n')
    claims_file.close()

    topics_text = topics_file.read()
    topics_list = topics_text.split('\n')
    topics_file.close()

    evidence_text = evidence_file.read()
    evidence_list = evidence_text.split('\n')
    evidence_file.close()

    claims_list = claims_list[1:]
    topics_list = topics_list[1:]
    evidence_list = evidence_list[1:]

    topic_to_articles = {}
    claim_version = {}

    for topic_row in topics_list:
        if len(topic_row) == 0:
            continue
        parts = topic_row.split('\t')
        topic = parts[0]
        aid = parts[2]

        if topic not in topic_to_articles.keys():
            topic_to_articles[topic] = set()
        topic_to_articles[topic].add(aid)

    for claim_row in claims_list:
        if len(claim_row) == 0:
            continue
        parts = claim_row.split('\t')
        corrected_claim = parts[1]
        original_claim = parts[2]
        topic = parts[0]

        original_claim = process_IBM_strings(original_claim)
        corrected_claim = process_IBM_strings(corrected_claim)

        if corrected_claim not in claim_version.keys():
            claim_set = set()
            claim_set.add(corrected_claim)
            claim_version[corrected_claim] = claim_set
        claim_version[corrected_claim].add(original_claim)

    new_evidence_row = ("topic\tclaim\tevidence\ttypes\tdocument\n")
    all_file_new.write(new_evidence_row)

    for evidence_row in evidence_list:
        if len(evidence_row) == 0:
            continue
        parts = evidence_row.split('\t')
        if len(parts) < 4:
            print("ERROR! NOT ENOUGH ELEMENTS")
            print("\t" + str(parts))
            continue

        topic = parts[0]
        corrected_claim = parts[1]
        evidence = parts[2]
        type = parts[3]

        corrected_claim = process_IBM_strings(corrected_claim)
        evidence = process_IBM_strings(evidence)

        article_list = topic_to_articles[topic]

        true_claim = ""
        doc = []

        for i in article_list:
            file_name = "clean_" + str(i)
            text_file_path = os.path.join(article_path, file_name + ".txt")

            if os.path.exists(text_file_path):
                text_file = open(text_file_path, 'r', encoding='utf-8')
                text = text_file.read()
                text_file.close()

                text = process_IBM_strings(text)

                index1 = text.find(evidence)
                if index1 >= 0:
                    for claim in claim_version[corrected_claim]:
                        index2 = text.find(claim)
                        if index2 >= 0:
                            doc.append(i)
                            true_claim = claim

        if len(doc) < 1:
            print("ERROR! COMPONENTS NOT FOUND IN " + str(article_list) +
                  "\n\t" + evidence +
                  "\n\t" + claim_version)

        new_doc = ""
        for num in doc:
            new_doc += str(num) + " "

        new_evidence_row = (topic + "\t" + true_claim + "\t" + evidence + "\t" +
                            type + "\t" + new_doc + "\n")
        all_file_new.write(new_evidence_row)
    all_file_new.close()

    print("DONE!")


def process_IBM_strings(string):
    """
    Format all the IBM string in the same way, creating a single string of lowercase characters
    :param string:
    :return:
    """
    parts = string.split()
    result = str(parts[0].lower())
    for part in parts[1:]:
        result += " " + str(part.lower())
    return result


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

    print()
    df1 = df[['source_ID', 'source_type']]
    column = 'source_type'
    df2 = df1.drop_duplicates()
    print(len(df2))
    print(df2[column].value_counts())


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

    dataset_name = 'IBM_CE_15'
    dataset_version = 'original_data'
    link_types = ['support']
    i = 1
    # find_IBM_claim_article(dataset_path)

    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)

    create_IBM_pickle(dataset_path, dataset_version, link_types, test=0.2, validation=0.1, reflexive=False)

    dataset_version = 'original_data_d100'

    for split in ['complete', 'train', 'test', 'validation']:
        print('_______________________')
        print(split)
        dataframe_path = os.path.join(dataset_path, 'pickles', dataset_version, split + '.pkl')
        print_dataframe_details(dataframe_path)
        print('_______________________')
        sys.stdout.flush()

    print('_______________________')
    print('_______________________')
    print('_______________________')

    highest = 0
    lowest = 0

    for split in ['complete', 'train', 'test', 'validation']:
        print(split)
        dataframe_path = os.path.join(dataset_path, 'pickles', dataset_version, split + '.pkl')
        df = pandas.read_pickle(dataframe_path)

        diff_l = {}
        diff_nl = {}

        for index, row in df.iterrows():
            s_index = int(row['source_ID'].split('_')[i])
            t_index = int(row['target_ID'].split('_')[i])

            difference = (s_index - t_index)

            if highest < difference:
                highest = difference
            if lowest > difference:
                lowest = difference

            if row['source_to_target']:
                voc = diff_l
            else:
                voc = diff_nl

            if difference in voc.keys():
                voc[difference] += 1
            else:
                voc[difference] = 1

        print()
        print()
        print(split)
        print("distance\tnot links\tlinks")
        for key in range(lowest, highest + 1):
            if key not in diff_nl.keys():
                diff_nl[key] = 0
            if key not in diff_l.keys():
                diff_l[key] = 0

            print(str(key) + "\t" + str(diff_nl[key]) + '\t' + str(diff_l[key]))

        sys.stdout.flush()