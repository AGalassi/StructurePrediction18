__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"

"""
output_units : (link classifier, relation classifier, source classifier, target classifier)
"""

dataset_info = {"AAEC_v2":      {"output_units": (2, 5, 3, 3),
                                 "min_text": 168,
                                 "min_prop": 72,
                                 "link_as_sum": [[0, 2], [1, 3, 4]],
                                 "categorical_prop": {'Premise': [1, 0, 0, ],
                                                      'Claim': [0, 1, 0, ],
                                                      'MajorClaim': [0, 0, 1],
                                                     },
                                 "categorical_link": {'supports': [1, 0, 0, 0, 0],
                                                      'inv_supports': [0, 1, 0, 0, 0],
                                                      'attacks': [0, 0, 1, 0, 0],
                                                      'inv_attacks': [0, 0, 0, 1, 0],
                                                      None: [0, 0, 0, 0, 1],
                                                     },
                                 "evaluation_headline_short": ("set\t" +
                                                               "F1 AVG all\tF1 AVG LP\tF1 Link\t" +
                                                               "F1 R AVG dir\tF1 R support\tF1 R attack\t" +
                                                                "F1 P AVG\t" +
                                                                "F1 P premise\tF1 P claim\tF1 P major claim\t" +
                                                                "F1 P avg\n\n")
                                },

                "cdcp_ACL17":   {"output_units": (2, 5, 5, 5),
                                 "min_text": 552,
                                 "min_prop": 153,
                                 "link_as_sum": [[0, 2], [1, 3, 4]],
                                 "categorical_prop": {'policy': [1, 0, 0, 0, 0],
                                                      'fact': [0, 1, 0, 0, 0],
                                                      'testimony': [0, 0, 1, 0, 0],
                                                      'value': [0, 0, 0, 1, 0],
                                                      'reference': [0, 0, 0, 0, 1],
                                                     },
                                 "categorical_link": {'reasons': [1, 0, 0, 0, 0],
                                                      'inv_reasons': [0, 1, 0, 0, 0],
                                                      'evidences': [0, 0, 1, 0, 0],
                                                      'inv_evidences': [0, 0, 0, 1, 0],
                                                      None: [0, 0, 0, 0, 1],
                                                     },
                                 "evaluation_headline_short": ("set\t"
                                                               "F1 AVG all\tF1 AVG LP\tF1 Link\t"
                                                               "F1 R AVG dir\tF1 R reason\tF1 R evidence\t" +
                                                               "F1 P AVG\t" +
                                                               "F1 P policy\tF1 P fact\tF1 P testimony\t" +
                                                               "F1 P value\tF1 P reference\tF1 P avg\n\n")
                                },

                "RCT":          {"output_units": (2, 5, 2, 2),
                                 "min_text": 2, # wrong, never measured
                                 "min_prop": 181,
                                 "link_as_sum": [[0, 2], [1, 3, 4]],
                                 "categorical_prop": {'Premise': [1, 0, ],
                                                      'Claim': [0, 1, ]
                                                     },
                                 "categorical_link": {'support': [1, 0, 0, 0, 0],
                                                      'inv_support': [0, 1, 0, 0, 0],
                                                      'attack': [0, 0, 1, 0, 0],
                                                      'inv_attack': [0, 0, 0, 1, 0],
                                                      None: [0, 0, 0, 0, 1],
                                                     },
                                 "evaluation_headline_short": ("set\t" +
                                                               "F1 AVG all\t" +
                                                               "F1 AVG LP\t" +
                                                               "F1 Link\t" +
                                                               "F1 R AVG dir\tF1 R support\tF1 R attack\t" +
                                                               "F1 P AVG\t" +
                                                               "F1 P premise\tF1 P claim\t" +
                                                               "F1 P avg\n\n")
                                },

                "DrInventor":   {"output_units": (2, 5, 3, 3),
                                 "min_text": 2, # wrong, never measured
                                 "min_prop": 181, # wrong, never measured
                                 "link_as_sum": [[0, 2], [1, 3, 4]],
                                 "categorical_prop": {'own_claim': [1, 0, 0,],
                                                      'background_claim': [0, 1, 0],
                                                      'data': [0, 0, 1],
                                                     },
                                 "categorical_link": {'supports': [1, 0, 0, 0, 0],
                                                      'inv_supports': [0, 1, 0, 0, 0],
                                                      'contradicts': [0, 0, 1, 0, 0],
                                                      'inv_contradicts': [0, 0, 0, 1, 0],
                                                      None: [0, 0, 0, 0, 1],
                                                     },
                                 "evaluation_headline_short": ("set\t" +
                                                               "F1 AVG all\tF1 AVG LP\tF1 Link\t" +
                                                               "F1 R AVG dir\tF1 R supports\tF1 R contradicts\t" +
                                                               "F1 P AVG\t" +
                                                               "F1 P own_claim\tF1 P background_claim\tF1 P data\t" +
                                                               "F1 P avg\n\n")
                                },

                }
