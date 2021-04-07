__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018-2020 Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.2.0"
__email__ = "a.galassi@unibo.it"

"""
Code to print the details of a dataframe from a specific corpus
"""

import os
import pandas
import json
import random
import sys
import ast
import numpy as np
import argparse


def print_dataframe_details(dataframe_path):
    df = pandas.read_pickle(dataframe_path)

    print(df.head())

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

    print("LIST OF DOCUMENT IDs")
    column = 'text_ID'
    print(column)
    print(list(df[column].drop_duplicates()))


def print_details(dataset_name, dataset_version):

    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)

    for split in ('train', 'test', 'validation', 'total'):
        print(split)
        dataframe_path = os.path.join(dataset_path, 'pickles', dataset_version, split + '.pkl')

        print_dataframe_details(dataframe_path)
        print('_______________________')
        print('_______________________')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create a new dataframe")

    parser.add_argument('-c', '--corpus',
                        choices=["rct", "drinv", "cdcp", "echr", "ukp"],
                        help="Corpus", default="cdcp")

    args = parser.parse_args()

    corpus = args.corpus

    if corpus.lower() == "rct":
        dataset_name = "RCT"
        dataset_version = 'total'
    elif corpus.lower() == "cdcp":
        dataset_name = 'cdcp_ACL17'
        dataset_version = 'new_3'
    elif corpus.lower() == "drinv":
        dataset_name = 'DrInventor'
        dataset_version = 'arg10'
    elif corpus.lower() == "ukp":
        dataset_name = 'AAEC_v2'
        dataset_version = 'new_2'
    else:
        print("Datset not yet supported")

    print_details(dataset_name, dataset_version)
