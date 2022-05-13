# Predicting-Binding-Energies-of-Astrochemically-Relevant-Molecules-via-Machine-Learning
Supporting repository to the article containing everything necessary to replicate and/or further expand on the work.

First download all six files from the repository and save them in the same folder. 

'GPRPredictAndValidate.py' is the main program. It is used to predict the BE of new molecules and validate the performance of the model.

To initiate the program install necessary modules e.g. RDKit https://www.rdkit.org/docs/Install.html 

When running the 'GPRPredictAndValidate.py' program in validation mode ('kfold') a numpy file ('GPRmono.npy' or 'GPRmulti.npy') is generated and saved in the folder, which includes the BE predicted by the model and the actual BE from the litterature. This file can be loaded into 'ParityPlot.py' to get fig. 2 from the article.

If changes to the 'Data_set_BEs.xlsx' file is made the 'SMILESmono.txt' and 'SMILESmulti.txt' have to be rewritten as well. This is done using the program 'SMILES_converter.py' which read the molecule name and convert to a SMILES-string and add it to a .txt file for further usage.
