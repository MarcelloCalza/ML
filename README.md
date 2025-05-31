# Project 2: Tree predictors for binary classification
Tree predictor + random forest for binary classification.

# run.py 
Runs a demonstration that executes everything using the Mushroom dataset.
# mushroom/tree.py
Contains the classes for the nodes (Node) and the tree predictors (DecisionTree).
# mushroom/forest.py
Contains the random forest class (RandomForest).
# mushroom/tuning.py
Contains the functions used for single level K-fold CV tuning for both the trees and the random forests.
# mushroom/diagnostics.py
Contains the functions used to execute empirical approximated bias/variance analysis.
# mushroom/criteria.py
Contains just the impurity functions.
# report/
Contains all the latex files of the report (and the report's pdf).
# data/
Contains some CSVs of the intermediate results of run.py that were saved.
# plots/
Contains plots PNGs produced by run.py through the plotting function in diagnostics.py.

The project is implemented using raw Python, the average time to build a tree should be around 6-7s, 
however CV tuning for the tree/random forests and random forests building can take long because more 
trees needs to be built. Also capacity parameter sweeping can take even longer depending on the values
of the capacity parameter.
