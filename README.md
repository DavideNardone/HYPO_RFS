# HYPO_RFS

HYPO_RFS is an exhaustive grid-search Ranking Feature Selection (RFS) approach for hyperparameter optimization. If you don't have any background on feature selection, please refer to the following [source](https://machinelearningmastery.com/an-introduction-to-feature-selection/).

A class of Feature Selection (FS) algorithm, the so called RFS, outputs certain information (e.g., weight matrix, score vector, etc.) such that is possible to exploit such elements as feature ranking information (e.g., sorting the RFS scores in a descending order such that: the higher the score, the more important the feature is). Some RFS algorithms are: Relief, Fisher, RFS, etc. Here for [more information](http://eprints.kku.edu.sa/170/1/feature_selection_for_classification.pdf) about these algorithms.

As you may know, most of the FS algorithms are ruled by several parameters, which in turn (whether changed) may affect the accuracy of a classification task. Therefore, it is important to tune such parameters in order to find the best combination of them which work better for a certain number of features (e.g., the first 30/500 ranked features).

The approach presented here, uses an exhaustive grid-search combined with a majority vote approach for tuning the hyper-parameters of RFS algorithms.

# Requirements

  - python 2.7
  - numpy
  - sklearn
  - skfeature ([download from here](https://github.com/jundongl/scikit-feature/tree/master/skfeature))
 
 # Usage
 
 Run `hypo_main.py` for a naive example. It provides the best combination for a single parameter.
 
 # Example
 
 In this example we tuned the *γ* parameter for the RFS approach.
 
![alt text](img/Relief.png "")

`Output:` <br>
Hyper-params. comb=(1e-15,) has minimum variance of 0.00183 <br>
Hyper-params. comb=(1e-10,) has minimum variance of 0.001837 <br>
Hyper-params. comb=(1e-08,) has minimum variance of 0.001837 <br>
Hyper-params. comb=(1e-05,) has minimum variance of 0.001837 <br>
Hyper-params. comb=(0.0001,) has minimum variance of 0.001837 <br>
Hyper-params. comb=(0.001,) has minimum variance of 0.00134 <br>
Hyper-params. comb=(0.01,) has minimum variance of 0.00668 <br>
Hyper-params. comb=(1,) has minimum variance of 0.00933 <br>
Hyper-params. comb=(5,) has minimum variance of 0.01163 <br>
**Hyper-params. comb=(10,) has minimum variance of 0.00985** <br>

Applying majority voting... <br>
Parameters set: (1e-15,) got 1.0 votes <br>
Parameters set: (1e-10,) got 1.0 votes <br>
Parameters set: (1e-08,) got 1.0 votes <br>
Parameters set: (1e-05,) got 1.0 votes <br>
Parameters set: (0.0001,) got 1.0 votes <br>
Parameters set: (0.001,) got 0.0 votes <br>
Parameters set: (0.01,) got 0.0 votes <br>
Parameters set: (1,) got 2.0 votes <br>
Parameters set: (5,) got 1.0 votes <br>
**Parameters set: (10,) got 98.0 votes** <br>

Best parameters set found on development set is: (10,) <br>

 # Authors

  Davide Nardone
  
  https://www.linkedin.com/in/davide-nardone-127428102/
  
# Contacts

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact me at: 
- **davide.nardone@live.it**
 
