# Welcome to my end of the year project - TRIDENT Neutrino Energy Prediction

In this code you will find all the information about the two models used to predict enegies, MLP and DeepSet.

Before running, make sure to configure btoh config files for running the MLP and the DeepSet.

### Install Requirements

'''
pip install -r requirements.txt
'''


Please have downloaded a file with the trident data. The name of the folder should be:

*trident_data*

In it should have:

*features.parquet*
*truth_minimal.parquet*

## Running the code

Training the MLP model:

'''
python3 training_mlp.py
python3 evaluate_mlp.py
'''

Training the Deep Sets model:

'''
python3 training_deep_sets.py
python3 evaluate_deep_sets.py
'''

## Running Multiple Things at Once

'''
python3 training_mlp.py && python3 evaluate_mlp.py
python3 training_deep_sets.py && python3 evaluate_deep_sets.py
'''

## Running Both Models at the samme time 

'''
python3 training_mlp.py && python3 training_deep_sets.py
python3 evaluate_mlp.py && python3 evaluate_deep_sets.py
'''

## Running Everything

'''
python training_mlp.py && python evaluate_mlp.py && \
python training_deep_sets.py && python evaluate_deep_sets.py
'''


