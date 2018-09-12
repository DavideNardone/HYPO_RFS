# HYPO_RFS

HYPO_RFS is a exhaustive grid-search Ranking Feature Selection (RFS) approach for hyperparameter optimization. If you don't have any background on feature selection, please refer to the following [source](https://machinelearningmastery.com/an-introduction-to-feature-selection/).

Feature Selection (FS) algorithms usually output several information and depending on the type the algorithm they might output some information regarding the relevance of the features retrieved. Such class of algorithm may be know as Ranking Feature Selection.
  
RFS algorithms output certain information such that it's possible to exploit it as *ranking* information (e.g., sorting the RFS scores in an descending order such that the higher the score, the more important the feature is). Some RFS alogorithm are: Relief, Fisher, RFS, etc. Here for [more information](http://eprints.kku.edu.sa/170/1/feature_selection_for_classification.pdf) about these algorithms.

As you may also know, most of the FS algorithms are ruled by several pararameters which in turn, whether changed, may provide different outputs. Therefore, it is importat to tune these parameter in order to find the best combination of them which work better for a certain number of features (e.g., the first 30/500 ranked features).

The algorithm presented here, uses a *grid-search* combined with a *majority vote* approach for tunining the hyper-parameters of the FS algorithm.


# Requirements

  - python 2.7
  - numpy
  - sklearn
  - skfeature ([download from here](https://github.com/jundongl/scikit-feature/tree/master/skfeature))
 
 # Usage
 
 Run `hypo_main.py` for a naive example. It provides the best combination for a single parameter.
 
 
 # Authors

  Davide Nardone
  
  https://www.linkedin.com/in/davide-nardone-127428102/
  
# Contacts

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@live.it**
 
