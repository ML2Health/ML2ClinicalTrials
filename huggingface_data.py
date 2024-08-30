# downloading data from huggingface

# account: https://huggingface.co/datasets/ML2Healthcare/ClinicalTrialDataset

from datasets import load_dataset
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = {}
    dataset = load_dataset('ML2Healthcare/ClinicalTrialDataset')
    dataset = dataset['train'].to_dict()
    for task, phase, type_, table in zip(dataset['task'], dataset['phase'], dataset['type'], dataset['data']):
        table = pd.DataFrame.from_dict(eval(table, {'nan': np.nan}))
        table_name = f"{task}_{phase}_{type_}"
        data[table_name] = table

