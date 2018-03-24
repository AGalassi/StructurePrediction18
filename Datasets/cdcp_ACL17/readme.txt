Cornell eRulemaking Corpus - CDCP

## 1. Overview ##################################

This dataset consists argument annotations on user comments about rule proposals regarding Consumer Debt Collection Practices by the Consumer Financial Protection Bureau crawled from an eRulemaking website, regulationroom.org, run by Cornell eRulemaking Initiative (CeRI).  

The annotation scheme is based on the argumentation model presented in "Toward Machine-assisted Participation in eRulemaking: An Argumentation Model of Evaluability" by Joonsuk Park, Cheryl Blake and Claire Cardie (ICAIL 2015) 

If you have any questions, please contact Joonsuk Park (park@joonsuk.org).


This Cornell eRulemaking Corpus - CDCP is made available under Open Database License whose full text can be found at http://opendatacommons.org/licenses/odbl/. Any rights in individual contents of the database are licensed under the Database Contents License whose text can be found http://opendatacommons.org/licenses/dbcl/



## 2. File Naming Conventions ##################################
- id.txt: text of a comment

- id.ann.json: argument structure annotation on the text

- id.txt.json, id.txt.xml: output of CoreNLP v3.6.0 
(https://github.com/stanfordnlp/CoreNLP/blob/master/README.md)

- id.txt.pipe: output of the PDTB-style End-to-End Discourse Parser. v2.0.2
(https://github.com/WING-NUS/pdtb-parser/commit/5ee603a9c718ad6cb5fb3f291ab201be0ab42b2c)



## 3. id.ann.json File Format ##################################

Here's a description of the annotation format using an example annotation.
Please refer to Park et al. (2015) for the full argumentation model.

Each file contains a map:
{

"prop_labels": ["fact", "value", "policy"], // the type of each of the propositions (3 propositions in this case) in the comment

"prop_offsets": [[0, 149], [149, 232], [232, 396]], // character offsets of each proposition

"reasons": %support%, // a list of [source_range, target], where source_range is a range of propositions serving as a reason for the target proposition

"evidences": %support%, // a list of [source_range, target], where source_range is a range of propositions serving as a piece of evidence for the target proposition

"url": {"17": "http://www.daily-protest.blogspot.com"} // URLs are replaced with a URL token in .txt files. "url" maps proposition numbers (e.g. "17") to the replaced URL (e.g. "http://www.daily-protest.blogspot.com")

}

where
%support% = 
[
[[1,1],2], // proposition 1 supports proposition 2
[[1,3],5], // propositions 1, 2, and 3 collectively support proposition 5
]



## 4. Publications ##################################

This dataset has been used in the following publication:

Vlad Niculae, Joonsuk Park and Claire Cardie. Argument Mining with Structured SVMs and RNNs. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL). July 2017.
(The code for the experiments is available at
https://github.com/vene/marseille)
