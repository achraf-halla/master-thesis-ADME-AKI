import os
import json
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen, QED
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

# --- paths ---
DEFAULT_SCALER_DIR = os.path.join("models", "scalers")
META_DIR = os.path.join("results", "pretraining")
META_PATH = os.path.join(META_DIR, "featurization_meta.json")
os.makedirs(DEFAULT_SCALER_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

ESSENTIAL_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "ExactMolWt": Descriptors.ExactMolWt,
    "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "Chi0": Descriptors.Chi0, "Chi1": Descriptors.Chi1,
    "Chi2n": Descriptors.Chi2n, "Chi3n": Descriptors.Chi3n, "Chi4n": Descriptors.Chi4n,
    "Chi2v": Descriptors.Chi2v, "Chi3v": Descriptors.Chi3v, "Chi4v": Descriptors.Chi4v,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    "Kappa1": Descriptors.Kappa1, "Kappa2": Descriptors.Kappa2, "Kappa3": Descriptors.Kappa3,
    "NumHAcceptors": rdMolDescriptors.CalcNumLipinskiHBA,
    "NumHDonors": rdMolDescriptors.CalcNumLipinskiHBD,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "RingCount": Descriptors.RingCount,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
    "NumHeterocycles": rdMolDescriptors.CalcNumHeterocycles,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms,
    "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms,
    "FormalCharge": lambda m: sum(a.GetFormalCharge() for a in m.GetAtoms()) if m else np.nan,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NHOHCount": Descriptors.NHOHCount,
    "NOCount": Descriptors.NOCount,
    "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds,
    "QED": QED.qed,
    "CrippenLogP": Crippen.MolLogP,
    "CrippenMR": Crippen.MolMR,
}

def compute_vsa_descriptors(mol):
    out = {}
    if mol is None:
        return out
    try:
        for i, v in enumerate(rdMolDescriptors.PEOE_VSA_(mol), 1):
            out[f"PEOE_VSA_{i}"] = v
        for i, v in enumerate(rdMolDescriptors.SMR_VSA_(mol), 1):
            out[f"SMR_VSA_{i}"] = v
        for i, v in enumerate(rdMolDescriptors.SlogP_VSA_(mol), 1):
            out[f"SlogP_VSA_{i}"] = v
    except Exception:
        pass
    return out

def compute_fast3d_descriptors(mol):
    keys = ["PMI1", "PMI2", "PMI3", "Asphericity", "Eccentricity",
            "RadiusOfGyration", "SpherocityIndex", "InertialShapeFactor"]
    out = {k: np.nan for k in keys}
    if mol is None:
        return out
    try:
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
        out["PMI1"], out["PMI2"], out["PMI3"] = rdMolDescriptors.CalcPrincipalMomentsOfInertia(mol)
        out["Asphericity"] = rdMolDescriptors.CalcAsphericity(mol)
        out["Eccentricity"] = rdMolDescriptors.CalcEccentricity(mol)
        out["RadiusOfGyration"] = rdMolDescriptors.CalcRadiusOfGyration(mol)
        out["SpherocityIndex"] = rdMolDescriptors.CalcSpherocityIndex(mol)
        out["InertialShapeFactor"] = rdMolDescriptors.CalcInertialShapeFactor(mol)
    except Exception:
        pass
    return out

def compute_descriptor_df(smiles_series, use_vsa=True, use_3d=False):
    rows = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi) if pd.notnull(smi) else None
        rec = {}
        for name, fn in ESSENTIAL_DESCRIPTOR_FUNCS.items():
            try:
                rec[name] = fn(mol) if mol else np.nan
            except Exception:
                rec[name] = np.nan
        try:
            if mol:
                mol_ch = Chem.AddHs(Chem.Mol(mol))
                AllChem.ComputeGasteigerCharges(mol_ch)
                charges = [float(a.GetProp('_GasteigerCharge')) for a in mol_ch.GetAtoms()
                           if a.HasProp('_GasteigerCharge')]
                if charges:
                    rec['GasteigerCharge_mean'] = float(np.mean(charges))
                    rec['GasteigerCharge_min'] = float(np.min(charges))
                    rec['GasteigerCharge_max'] = float(np.max(charges))
                else:
                    rec['GasteigerCharge_mean'] = rec['GasteigerCharge_min'] = rec['GasteigerCharge_max'] = np.nan
            else:
                rec['GasteigerCharge_mean'] = rec['GasteigerCharge_min'] = rec['GasteigerCharge_max'] = np.nan
        except Exception:
            rec['GasteigerCharge_mean'] = rec['GasteigerCharge_min'] = rec['GasteigerCharge_max'] = np.nan

        if use_vsa:
            try:
                rec.update(compute_vsa_descriptors(mol))
            except Exception:
                pass
        if use_3d and mol:
            try:
                rec.update(compute_fast3d_descriptors(Chem.AddHs(Chem.Mol(mol))))
            except Exception:
                pass
        rows.append(rec)
    return pd.DataFrame(rows, index=getattr(smiles_series, "index", None))


def _choose_imputer(df, missing_thresh=0.10, knn_neighbors=5):
    high_missing_cols = (df.isnull().mean() > missing_thresh).sum()
    if high_missing_cols > 0:
        imp = KNNImputer(n_neighbors=knn_neighbors)
        imputer_type = "knn"
    else:
        imp = SimpleImputer(strategy="median")
        imputer_type = "median"
    return imp, imputer_type

def fit_descriptor_pipeline(df_train,
                            missing_thresh=0.10,
                            skew_thresh=1.0,
                            indicator_thresh=0.01,
                            knn_neighbors=5):
    """
    Fit imputer + per-column scaler decisions on df_train (DataFrame of raw descriptors).
    Saves imputer and scalers to DEFAULT_SCALER_DIR and writes meta to META_PATH.
    Returns meta dict.
    """
    df = df_train.copy()
    keep_mask = ~((df.isna().all()) | (df.nunique(dropna=True) <= 1))
    df = df.loc[:, keep_mask]
    cols = list(df.columns)

    imputer, imputer_type = _choose_imputer(df, missing_thresh, knn_neighbors)
    imputed = pd.DataFrame(imputer.fit_transform(df), columns=cols, index=df.index)

    scalers = {}
    scaler_types = {}
    for c in cols:
        series = imputed[c].astype(float)
        skew = float(series.skew())
        if abs(skew) > skew_thresh:
            sc = RobustScaler()
            stype = "robust"
        else:
            sc = StandardScaler()
            stype = "standard"
        sc.fit(series.values.reshape(-1, 1))
        scalers[c] = sc
        scaler_types[c] = {"type": stype, "skew": skew}

    missing_rates = df.isnull().mean()
    indicator_cols = [c for c in cols if missing_rates[c] >= indicator_thresh]

    imputer_fname = os.path.join(DEFAULT_SCALER_DIR, f"desc_imputer_{imputer_type}.pkl")
    scalers_fname = os.path.join(DEFAULT_SCALER_DIR, "desc_scalers.pkl")
    joblib.dump(imputer, imputer_fname)
    joblib.dump(scalers, scalers_fname)

    meta = {
        "cols": cols,
        "indicator_cols": indicator_cols,
        "imputer": {"type": imputer_type, "path": imputer_fname},
        "scalers": {"path": scalers_fname, "per_column": scaler_types},
        "missing_thresh": float(missing_thresh),
        "skew_thresh": float(skew_thresh),
        "indicator_thresh": float(indicator_thresh),
    }
    meta["descriptor_dim"] = len(cols) + len(indicator_cols)
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta

def transform_descriptors(df, meta=None):
    """
    Align df (DataFrame of raw descriptors keyed by column names), impute, scale, append missing indicators.
    Returns np.array shape (n_samples, descriptor_dim).
    """
    if meta is None:
        with open(META_PATH, "r") as fh:
            meta = json.load(fh)
    cols = meta["cols"]
    indicator_cols = meta.get("indicator_cols", [])

    imputer = joblib.load(meta["imputer"]["path"])
    scalers = joblib.load(meta["scalers"]["path"])

    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            df2[c] = np.nan
    df2 = df2[cols]

    ind_arrays = []
    for c in indicator_cols:
        ind = df2[c].isnull().astype(int).values.reshape(-1)
        ind_arrays.append(ind)

    imputed = pd.DataFrame(imputer.transform(df2), columns=cols, index=df2.index)

    out_cols = []
    for c in cols:
        sc = scalers[c]
        col_scaled = sc.transform(imputed[[c]]).reshape(-1)
        out_cols.append(col_scaled)
    X_numeric = np.vstack(out_cols).T 

    if ind_arrays:
        X_ind = np.vstack(ind_arrays).T 
        X = np.concatenate([X_numeric, X_ind], axis=1)
    else:
        X = X_numeric

    return X

def load_featurization_meta(path=META_PATH):
    with open(path, "r") as fh:
        return json.load(fh)

def compute_and_transform(smiles_series, fit_meta=None, use_vsa=True, use_3d=True):
    raw = compute_descriptor_df(smiles_series, use_vsa=use_vsa, use_3d=use_3d)
    if fit_meta is None:
        meta = load_featurization_meta()
    else:
        meta = fit_meta
    return transform_descriptors(raw, meta)


def get_atom_features(a):
    return [
        a.GetAtomicNum(), a.GetDegree(),
        int(a.GetHybridization()), int(a.GetIsAromatic()),
        a.GetFormalCharge(), int(a.IsInRing()),
    ]

def get_bond_features(b):
    return [
        b.GetBondTypeAsDouble(), int(b.GetIsConjugated()),
        int(b.IsInRing()), int(b.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
    ]
from sklearn.preprocessing import StandardScaler
import joblib, os, numpy as np, pandas as pd

meta = load_featurization_meta()  
os.makedirs("results/pretraining", exist_ok=True)

def mol_graph_feats(series):
    A, B = [], []
    for smi in series:
        mol = Chem.MolFromSmiles(smi) if pd.notnull(smi) else None
        if mol:
            A.append([get_atom_features(a) for a in mol.GetAtoms()])
            B.append([get_bond_features(b) for b in mol.GetBonds()])
        else:
            A.append([]); B.append([])
    return A, B

processed = {}
targ_scalers = {}
tasks = sorted(train_df['task'].unique())

for task in tasks:
    tr = train_df[train_df['task'] == task].reset_index(drop=True)
    vl = val_df[val_df['task'] == task].reset_index(drop=True)
    te = test_df[test_df['task'] == task].reset_index(drop=True)

    tr_X = compute_and_transform(tr['Drug'], fit_meta=meta, use_vsa=True, use_3d=True)
    vl_X = compute_and_transform(vl['Drug'], fit_meta=meta, use_vsa=True, use_3d=True)
    te_X = compute_and_transform(te['Drug'], fit_meta=meta, use_vsa=True, use_3d=True)

    if tr.loc[0, 'task_type'] == 'regression':
        ts = StandardScaler().fit(tr[['Y']])
        targ_scalers[task] = ts
        tr_Y = ts.transform(tr[['Y']]).ravel()
        vl_Y = ts.transform(vl[['Y']]).ravel()
        te_Y = ts.transform(te[['Y']]).ravel()
    else:
        tr_Y, vl_Y, te_Y = tr['Y'].values, vl['Y'].values, te['Y'].values

    tr_A, tr_B = mol_graph_feats(tr['Drug'])
    vl_A, vl_B = mol_graph_feats(vl['Drug'])
    te_A, te_B = mol_graph_feats(te['Drug'])

    processed[task] = {
        'train': {'desc': tr_X, 'Y': tr_Y, 'atoms': tr_A, 'bonds': tr_B, 'smiles': tr['Drug'].values},
        'val':   {'desc': vl_X, 'Y': vl_Y, 'atoms': vl_A, 'bonds': vl_B, 'smiles': vl['Drug'].values},
        'test':  {'desc': te_X, 'Y': te_Y, 'atoms': te_A, 'bonds': te_B, 'smiles': te['Drug'].values},
    }



