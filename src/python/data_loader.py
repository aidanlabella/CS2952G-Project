#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np
import pandas as pd
import tensorflow as tf

def load_data():
    iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
    if iskaggle:
        # get the data path from within kaggle
        path = Path('/kaggle/input/stanford-ribonanza-rna-folding')
    else:
        # get the data path from local machine
        # you gotta make sure the data is actually there first, bc it is not committed to git
        path = Path('..data/')

    train = pd.read_csv(path/'train_data.csv')
    dms = train[train['experiment_type'] == 'DMS_MaP']
    ta3 = train[train['experiment_type'] == '2A3_MaP']
    del train

    preprocessed_dms = preprocess(dms)
    preprocessed_ta3 = preprocess(ta3)
    del dms, ta3
    
    return preprocessed_dms, preprocessed_ta3

def pad_seq(sequence, max_length=457, padding_value="Z"):
    padding_length = max_length - len(sequence)
    padded_sequence = list(sequence) + [padding_value] * padding_length
    return padded_sequence

def pad_react(reactivities, max_length=457, padding_value=0):
    padding_length = max_length - len(reactivities)
    padded_reactivities = reactivities + [padding_value] * padding_length
    return padded_reactivities

def one_hot_nuc(nucleotide):
    encoding = {
        "A": [1,0,0,0,0],
        "C": [0,1,0,0,0],
        "G": [0,0,1,0,0],
        "U": [0,0,0,1,0],
        "Z": [0,0,0,0,1]
    }
    return encoding[nucleotide]

def one_hot_seq(sequence):
    return [one_hot_nuc(n) for n in sequence]

def preprocess(data):
    # remove unnecessary columns
    data = data.drop(columns=data.filter(like='error').columns,axis=1)
    data = data.drop(['sequence_id', 'experiment_type', 'dataset_name', 'reads', 'signal_to_noise', 'SN_filter'],axis=1)

    # fill in null reactivities and clip to [0,1]
    data = data.fillna(0)
    reactivity_cols = data.filter(like='reactivity').columns
    data[reactivity_cols] = data[reactivity_cols].clip(lower=0,upper=1)

    # concatenate all reactivity column values to a single list
    data['reactivity'] = data[reactivity_cols].values.tolist()
    data = data.drop(columns=data.filter(like='reactivity_').columns,axis=1)

    # pad sequence and reactivities; one hot encode sequence
    data['sequence'] = data['sequence'].apply(pad_seq)
    data['sequence'] = data['sequence'].apply(one_hot_seq)
    data['sequence'] = data['sequence'].apply(lambda x: np.array(x, np.float32))
    data['reactivity'] = data['reactivity'].apply(pad_react)
    data['reactivity'] = data['reactivity'].apply(lambda x: np.array(x, np.float32))
    
    # convert everything to tensors
    data['sequence'] = data['sequence'].apply(tf.convert_to_tensor)
    data['reactivity'] = data['reactivity'].apply(tf.convert_to_tensor)
    
    return data
