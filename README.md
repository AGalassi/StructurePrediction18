# ResAttArg: Residual Attentive Deep Netoworks for Argument Structure Prediction.

Use of residual deep networks, ensemble learning, and attention for Argument Structure Prediction.

This code can be used to train a set of neural networks to jointly perform Link Prediction, Relation Classification, and Component Classification on Argument Mining corpora.
Currently, 4 corpora are supported:
- [CDCP](https://facultystaff.richmond.edu/~jpark/) (version with transitive closure and no nested proposition)
- [AbstRCT](https://gitlab.com/tomaye/abstrct/)
- [UKP Persuasive Essays v2](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp)
- [DrInventor](https://github.com/anlausch/sciarg_resource_analysis) (with some adaptation)
- [Scidtb](http://scientmin.taln.upf.edu/argmin/scidtb_argmin_annotations.tgz)

**Such corpora are not included in this repository. Only mock files are present so as to indicate where to place the documents to run the code.**

## Results and citing

The results of the ResAttArg architecture (net_11 in the code) and the new results of the ResAtt architecture 
(net_7 in the code) are available as preprint [Multi-Task Attentive Residual Networks for Argument Mining](https://arxiv.org/abs/2102.12227).
Such preprint is available in the publication folder.

The first results of the ResAtt architectures have been published in [Argumentative Link Prediction using Residual Networks and Multi-Objective Learning](https://www.aclweb.org/anthology/W18-5201).
Such models, along with the paper, are available in the Publications folder.


Please, if you use any of this material, cite our work both as:
```
@article{Galassi2021
  author    = {Andrea Galassi and
               Marco Lippi and
               Paolo Torroni},
  title     = {Multi-Task Attentive Residual Networks for Argument Mining},
  journal   = {CoRR},
  volume    = {abs/2102.12227},
  year      = {2021},
  url       = {https://arxiv.org/abs/2102.12227},
  archivePrefix = {arXiv},
  eprint    = {2102.12227},
}
```
and
```
@inproceedings{galassi-etal-2018-argumentative,
  title = "Argumentative Link Prediction using Residual Networks and Multi-Objective Learning",
  author = "Galassi, Andrea  and Lippi, Marco  and Torroni, Paolo",
  booktitle = "Proceedings of the 5th Workshop on Argument Mining",
  month = nov,
  year = "2018",
  address = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/W18-5201",
  doi = "10.18653/v1/W18-5201",
  pages = "1--10",
}
```

## Pipeline and other files

To use this framework, follow this pipeline, using the -c option to specify the desired dataset (cdcp, rct, ukp, drinv).
- dataframe_creator.py contains functions to process the textual and annotation files into dataframes.
- glove_loader.py contains functions to tokenize words and create a file with pre-trained embeddings which are smaller than the original glove file.
- embedder.py contains functions to map each string of the dataframe into a sequence of numbers, according to word positions in the glove file.
- training.py contains functions to perform the training. The hyper-parameters are embedded in the code. Any change requires manually modify the "routine" functions.
- evaluate_net.py contains functions to evaluate an already trained network. It offers additional options, among which the option -t to perform the token-wise evaluation.

Out of the pipeline:
- print_dataset_details.py prints details regarding a dataset: statistics about the classes and the lists of the document ids for each split
- networks.py contains neural network models
- training_utils.py contains custom functions that will be used during the training

The GloVe vocabulary file, required for the use of the framework, is not included in this repository. Simply download it from the GloVe website and add it to the working directory. The name of the file must be 'glove.840B.300d.txt'.
