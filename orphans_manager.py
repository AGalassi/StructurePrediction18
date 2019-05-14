__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"

"""
Honestly I don't remember the purpose for this. I guess it is useful to find where an orphan belongs. Maybe.
"""


import pandas
import os
import numpy as np

if __name__ == '__main__':
    dataset_name = 'cdcp_ACL17'
    dataset_path = os.path.join(os.getcwd(),'Datasets', dataset_name)
    pickles_path = os.path.join(os.path.join(dataset_path, 'pickles', 'good'))
    dataframe_path = os.path.join(pickles_path, 'total.pkl')
    # vocabulary_source_path = os.path.join(os.getcwd(), 'glove.840B.300d.txt')
    glove_path = os.path.join(dataset_path, 'glove')

    df = pandas.read_pickle(dataframe_path)

    propositions = df['source_proposition'].drop_duplicates()

    orphans_path = os.path.join(glove_path, 'glove.orphans.txt')
    vocabulary_path = os.path.join(glove_path, 'glove.vocabulary.txt')
    orphans_log_path = os.path.join(glove_path, 'glove.orphans.log.txt')
    orphans_log_file = open(orphans_log_path, 'w')
    orphans_file = open(orphans_path, 'r')

    for line in orphans_file:
        if len(line) > 0:
            line = line.split()[0]
            orphans_log_file.write(line + '\n\n')

            for proposition in propositions:
                if line in proposition:
                    orphans_log_file.write(proposition + '\n\n')

            orphans_log_file.write('\n\n\n')

    orphans_file.close()
    orphans_log_file.close()
    print('Finished')

