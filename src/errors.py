import pandas as pd
import numpy as np

def per_molecule_errors(pred_csv):
    """pred_csv expected columns: SMILES, task, pred, true"""
    df = pd.read_csv(pred_csv)
    df['abs_err'] = (df['pred'] - df['true']).abs()
    # aggregate by SMILES
    agg = df.groupby('SMILES').agg({'abs_err': ['mean','max'], 'task': lambda x: list(x.unique())})
    agg.columns = ['err_mean','err_max','tasks']
    return agg.reset_index().sort_values('err_mean', ascending=False)

def error_by_scaffold(agg_df, scaffold_map):
    """scaffold_map: dict SMILES->scaffold (or Series)"""
    df = agg_df.merge(scaffold_map.rename('scaffold'), left_on='SMILES', right_index=True)
    return df.groupby('scaffold')['err_mean'].describe()
