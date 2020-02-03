import numpy as np

def get_sequence_dataset(data, multitask_attr=""):
    data = data.copy()
    patient_seqs = []
    outcome_seqs = []
    for patient_num in data['patient_nbr'].unique():
        patient_data = data[data['patient_nbr']==patient_num]
        patient_data = patient_data.sort_values(by=['encounter_id'])
        if multitask_attr != "":
            patient_outcome = patient_data[['time_in_hospital', multitask_attr]].to_numpy()
        else:
            patient_outcome = patient_data['time_in_hospital'].to_numpy()
        patient_data = patient_data.drop(['patient_nbr', 'encounter_id'], axis=1)
        patient_data = patient_data.to_numpy()
        outcome_seqs.append(patient_outcome[1:].tolist())
        patient_seqs.append(patient_data[:-1].tolist())
    return patient_seqs, outcome_seqs

def get_oneshot_dataset(data, multitask_attr=""):
    x, y = get_sequence_dataset(data, multitask_attr)
    x = [pat for seq in x for pat in seq]
    y = [outcome for seq in y for outcome in seq]
    return x, y

""" TBC
def transform_sequence_dataset(patient_seqs, outcome_seqs, seq_length):
    prepared_pat_seqs = []
    prepared_out_seqs = []
    for pat_seq, out_seq in zip(patient_seqs, outcome_seqs):
"""

from transformers import *
if __name__ == '__main__':
    x = main()
    x, y = get_oneshot_dataset(x, 'num_medications')
    print(x[:10])
    print(y[:10])
    #print(get_sequence_dataset(x))
    #print(get_oneshot_dataset(x))
