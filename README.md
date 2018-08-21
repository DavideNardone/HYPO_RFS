# HYPO_RFS

HYPO_RFS is an algorithm for performing exhaustive grid-search approach for hyper-parameters optimization on Ranking Feature Selection (RFS) approach...

If you don't have any background on feature selection, please refer to the following [source](https://machinelearningmastery.com/an-introduction-to-feature-selection/).

Most Feature Selection (FS) algorithms output different information. Most of them, also provide information about the discriminative power of the feature selected, which can be exploit for *ranking* the retrieved feature, that is, the most discriminative feature are placed ahead to the others (e.g., sort the feature scores in an ascending order according to the feature scores). You can find more information about this type of algorithm [here](http://blog.datadive.net/selecting-good-features-part-ii-linear-models-and-regularization/).

As you may also know, most of the FS algorithm are ruled by several pararameters which in turn, whether changed, may provide different outputs. Therefore, it is importat to tune these parameter in order to find the best combination of them which work better for a certain number of features (e.g., the first 30/500 ranked features).

The algorithm presented here, uses a *grid-search* combined with a *majority vote* approach for tunining the hyper-parameters of the FS algorithm.


# Requirements

  - python 2.7
  - numpy
  - sklearn
  - skfeature ([download from here](https://github.com/jundongl/scikit-feature/tree/master/skfeature))
 
 # Usage
 
 Run `hypo_main.py` for a naive example. It provides the best combination for a single parameters.
 
 
 # Authors

  Davide Nardone
  
  https://www.linkedin.com/in/davide-nardone-127428102/
  
# Contacts

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@live.it**
 
