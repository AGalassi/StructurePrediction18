# StructurePrediction18

Use of residual deep networks, ensemble learning, and attention for Argument Structure Prediction.

This code can be used to train a set of neural networks to jointly perform Link Prediction, Relation Classification, and Component Classification on Argument Mining corpora.
Currently 4 corpora are supported:
- [CDCP](https://facultystaff.richmond.edu/~jpark/)
- [AbstRCT](https://gitlab.com/tomaye/abstrct/)
- [UKP Persuasive Essays v2](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp)
- [DrInventor](https://github.com/anlausch/sciarg_resource_analysis) (with some adaptation)


## Results and citing

The results of one of the first architecture implemented in this repository have been published in [Argumentative Link Prediction using Residual Networks and Multi-Objective Learning](https://www.aclweb.org/anthology/W18-5201). Such models (called net_7 in the code), along with the paper, are available in the Publications folder.

The results with the new models have not been published yet.


Please, if you use any of this material, cite our work as:
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
To use this framework, follow this pipeline:
- dataframe_creator.py contains functions to process the textual and annotation files into dataframes
- glove_loader.py contains functions to tokenize words and create a file with pre-trained embeddings which are smaller than the original glove file
- embedder.py contains functions to map each string of the dataframe into a sequence of numbers, according to word positions in the glove file
- training.py contains functions to perform the training

Out of the pipeline:
- networks.py contains neural network models
- training_utils.py contains custom function that will be used during the training
- evaluate_net.py contains functions to evaluate an already trained network

