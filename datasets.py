import numpy as np

def get_sequence_dataset(data):
    data = data.copy()
    patient_seqs = []
    outcome_seqs = []
    for patient_num in data['patient_nbr'].unique():
        patient_data = data[data['patient_nbr']==patient_num]
        patient_data = patient_data.sort_values(by=['encounter_id'])
        patient_outcome = patient_data['time_in_hospital'].to_numpy()
        patient_data = patient_data.drop(['patient_nbr', 'encounter_id'], axis=1)
        patient_data = patient_data.to_numpy()
        outcome_seqs.append(patient_outcome[1:].tolist())
        patient_seqs.append(patient_data[:-1].tolist())
    return patient_seqs, outcome_seqs

def get_oneshot_dataset(data):
    x, y = get_sequence_dataset(data)
    x = [pat for seq in x for pat in seq]
    y = [outcome for seq in y for outcome in seq]
    return x, y

from transformers import *
if __name__ == '__main__':
    x = main()
    print(get_sequence_dataset(x))
    print(get_oneshot_dataset(x))
