import os, json, joblib, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen, QED, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold

DEFAULT_SCALER_DIR = os.path.join("models", "scalers")
META_DIR = os.path.join("results", "pretraining")
META_PATH = os.path.join(META_DIR, "featurization_meta.json")
os.makedirs(DEFAULT_SCALER_DIR, exist_ok=True); os.makedirs(META_DIR, exist_ok=True)

CORE_DESCRIPTORS = {
    "MolWt", "HeavyAtomCount", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3"
}

ESSENTIAL_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt, "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "ExactMolWt": Descriptors.ExactMolWt, "MolLogP": Descriptors.MolLogP,
    "MolMR": Descriptors.MolMR, "TPSA": rdMolDescriptors.CalcTPSA,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA, "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT, "Chi0": Descriptors.Chi0, "Chi1": Descriptors.Chi1,
    "Chi2n": Descriptors.Chi2n, "Chi3n": Descriptors.Chi3n, "Chi4n": Descriptors.Chi4n,
    "Chi2v": Descriptors.Chi2v, "Chi3v": Descriptors.Chi3v, "Chi4v": Descriptors.Chi4v,
    "HallKierAlpha": Descriptors.HallKierAlpha, "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2, "Kappa3": Descriptors.Kappa3,
    "NumHAcceptors": rdMolDescriptors.CalcNumLipinskiHBA,
    "NumHDonors": rdMolDescriptors.CalcNumLipinskiHBD,
    "NumRotatableBonds": Descriptors.NumRotatableBonds, "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "RingCount": Descriptors.RingCount, "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings, "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
    "NumHeterocycles": rdMolDescriptors.CalcNumHeterocycles, "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms, "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms,
    "FormalCharge": lambda m: sum(a.GetFormalCharge() for a in m.GetAtoms()) if m else np.nan,
    "NumValenceElectrons": Descriptors.NumValenceElectrons, "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "NHOHCount": Descriptors.NHOHCount, "NOCount": Descriptors.NOCount,
    "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds, "QED": QED.qed,
    "CrippenLogP": Crippen.MolLogP, "CrippenMR": Crippen.MolMR,
}

def compute_vsa_descriptors(mol):
    out = {}
    if mol is None: return out
    try:
        for i, v in enumerate(rdMolDescriptors.PEOE_VSA_(mol), 1): out[f"PEOE_VSA_{i}"] = v
        for i, v in enumerate(rdMolDescriptors.SMR_VSA_(mol), 1): out[f"SMR_VSA_{i}"] = v
        for i, v in enumerate(rdMolDescriptors.SlogP_VSA_(mol), 1): out[f"SlogP_VSA_{i}"] = v
    except Exception:
        pass
    return out

def compute_fast3d_descriptors(mol):
    keys = ["PMI1","PMI2","PMI3","Asphericity","Eccentricity","RadiusOfGyration","SpherocityIndex","InertialShapeFactor"]
    out = {k: np.nan for k in keys}
    if mol is None: return out
    try:
        if mol.GetNumConformers() == 0: AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
        out["PMI1"], out["PMI2"], out["PMI3"] = rdMolDescriptors.CalcPrincipalMomentsOfInertia(mol)
        out["Asphericity"] = rdMolDescriptors.CalcAsphericity(mol)
        out["Eccentricity"] = rdMolDescriptors.CalcEccentricity(mol)
        out["RadiusOfGyration"] = rdMolDescriptors.CalcRadiusOfGyration(mol)
        out["SpherocityIndex"] = rdMolDescriptors.CalcSpherocityIndex(mol)
        out["InertialShapeFactor"] = rdMolDescriptors.CalcInertialShapeFactor(mol)
    except Exception:
        pass
    return out

def _maccs_bits(mol):
    try:
        fp = MACCSkeys.GenMACCSKeys(mol); return [int(fp.GetBit(i)) for i in range(1, 167)]
    except Exception:
        return [0]*166

def _morgan_stats(mol, nBits=2048, radius=2):
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits); on = fp.GetNumOnBits()
        return {'morgan_on_bits': int(on), 'morgan_density': float(on)/float(nBits)}
    except Exception:
        return {'morgan_on_bits': 0, 'morgan_density': 0.0}

def _hetero_counts(mol):
    if mol is None: return {'N_count':0,'O_count':0,'S_count':0,'F_count':0,'Cl_count':0,'Br_count':0,'P_count':0,'I_count':0}
    cnt = {'N':0,'O':0,'S':0,'F':0,'Cl':0,'Br':0,'P':0,'I':0}
    for a in mol.GetAtoms():
        s = a.GetSymbol();
        if s in cnt: cnt[s] += 1
    return {'N_count':cnt['N'],'O_count':cnt['O'],'S_count':cnt['S'],'F_count':cnt['F'],'Cl_count':cnt['Cl'],'Br_count':cnt['Br'],'P_count':cnt['P'],'I_count':cnt['I']}

def _scaffold_size(mol):
    try:
        smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else None
        sm = Chem.MolFromSmiles(smi) if smi else None
        return (sm.GetNumAtoms() if sm is not None else 0)
    except Exception:
        return 0

def _ring_stats(mol):
    if mol is None: return {'max_ring_size':0,'mean_ring_size':0.0,'n_rings':0}
    ssr = mol.GetRingInfo().AtomRings()
    if not ssr: return {'max_ring_size':0,'mean_ring_size':0.0,'n_rings':0}
    sizes = [len(r) for r in ssr]; return {'max_ring_size': int(max(sizes)), 'mean_ring_size': float(sum(sizes)/len(sizes)), 'n_rings': int(len(sizes))}

def _stereo_centers(mol):
    try: return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    except Exception: return 0

def compute_descriptor_df_enhanced(smiles_series, use_vsa=True, use_3d=False):
    rows = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi) if pd.notnull(smi) else None
        rec = {}
        for name, fn in ESSENTIAL_DESCRIPTOR_FUNCS.items():
            try: rec[name] = fn(mol) if mol else np.nan
            except Exception: rec[name] = np.nan
        try:
            if mol:
                mol_ch = Chem.AddHs(Chem.Mol(mol)); AllChem.ComputeGasteigerCharges(mol_ch)
                charges = [float(a.GetProp('_GasteigerCharge')) for a in mol_ch.GetAtoms() if a.HasProp('_GasteigerCharge')]
                rec['GasteigerCharge_mean'] = float(np.mean(charges)) if charges else np.nan
                rec['GasteigerCharge_min'] = float(np.min(charges)) if charges else np.nan
                rec['GasteigerCharge_max'] = float(np.max(charges)) if charges else np.nan
            else:
                rec['GasteigerCharge_mean'] = rec['GasteigerCharge_min'] = rec['GasteigerCharge_max'] = np.nan
        except Exception:
            rec['GasteigerCharge_mean'] = rec['GasteigerCharge_min'] = rec['GasteigerCharge_max'] = np.nan
        if use_vsa:
            try: rec.update(compute_vsa_descriptors(mol))
            except Exception: pass
        if use_3d and mol:
            try: rec.update(compute_fast3d_descriptors(Chem.AddHs(Chem.Mol(mol))))
            except Exception: pass
        maccs = _maccs_bits(mol)
        for i, b in enumerate(maccs, 1): rec[f"MACCS_{i}"] = b
        rec.update(_morgan_stats(mol, nBits=2048, radius=2))
        rec.update(_hetero_counts(mol))
        try:
            ha = rec.get('HeavyAtomCount', np.nan)
            if mol is None or not ha: rec['frac_aromatic_atoms'] = np.nan
            else: rec['frac_aromatic_atoms'] = float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())) / float(ha)
        except Exception: rec['frac_aromatic_atoms'] = np.nan
        rec['n_stereo_centers'] = _stereo_centers(mol)
        rec['scaffold_size'] = _scaffold_size(mol)
        rec.update(_ring_stats(mol))
        try: rec['tpsa_per_heavy'] = (rec.get('TPSA', np.nan) / float(rec.get('HeavyAtomCount'))) if rec.get('HeavyAtomCount') else np.nan
        except Exception: rec['tpsa_per_heavy'] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows, index=getattr(smiles_series, "index", None))

def _choose_imputer(df, missing_thresh=0.10, knn_neighbors=5):
    high_missing_cols = (df.isnull().mean() > missing_thresh).sum()
    if high_missing_cols > 0: return KNNImputer(n_neighbors=knn_neighbors), "knn"
    return SimpleImputer(strategy="median"), "median"

def fit_descriptor_pipeline(df_train,
                            missing_thresh=0.15,      
                            skew_thresh=2.0,          
                            indicator_thresh=0.05,    
                            zero_ratio_thresh=0.98,   
                            var_thresh=1e-6,          
                            corr_thresh=0.98,        
                            winsor_q=(0.005, 0.995), 
                            knn_neighbors=5,
                            min_features=20):         
    """
    Fit imputer + per-column scaler decisions, remove near-constant cols, drop very-highly-correlated columns,
    winsorize skewed columns, save scalers/imputer and meta.

    NEW: Ensures minimum number of features are kept by adjusting thresholds if needed.
    """
    df = df_train.copy()
    print(f"Starting with {len(df.columns)} descriptors")

    before_empty = len(df.columns)
    df = df.loc[:, ~(df.isna().all() | (df.nunique(dropna=True) <= 1))]
    print(f"After removing empty/constant: {len(df.columns)} (-{before_empty - len(df.columns)})")

    all_input_cols = list(df.columns)

    zero_ratio = (df == 0).mean()
    drop_zero_candidates = list(zero_ratio[zero_ratio >= zero_ratio_thresh].index)
    drop_zero = [col for col in drop_zero_candidates if col not in CORE_DESCRIPTORS]

    if drop_zero:
        df = df.drop(columns=drop_zero)
        print(f"After removing high-zero columns: {len(df.columns)} (-{len(drop_zero)})")

    imputer, imputer_type = _choose_imputer(df, missing_thresh, knn_neighbors)
    cols = list(df.columns)
    imputed = pd.DataFrame(imputer.fit_transform(df), columns=cols, index=df.index)

    vt = VarianceThreshold(threshold=var_thresh)
    try:
        vt.fit(imputed.fillna(0).values)
        keep_mask = vt.get_support()
        variance_kept = [c for c, k in zip(cols, keep_mask) if k]
        variance_dropped = [c for c in cols if c not in variance_kept]

        if len(variance_kept) < min_features:
            core_to_rescue = [c for c in variance_dropped if c in CORE_DESCRIPTORS]
            variance_kept.extend(core_to_rescue)
            variance_dropped = [c for c in variance_dropped if c not in core_to_rescue]
            print(f"Rescued {len(core_to_rescue)} core descriptors from variance filter")

    except Exception:
        variance_kept = cols
        variance_dropped = []

    print(f"After variance filter: {len(variance_kept)} (-{len(variance_dropped)})")
    kept = imputed[variance_kept].copy()

    winsor_caps = {}
    q_low, q_high = winsor_q
    skew_vals = kept.apply(lambda s: float(s.skew()) if len(s.dropna()) > 1 else 0.0)
    winsorize_cols = [c for c in kept.columns if abs(skew_vals.get(c, 0.0)) > skew_thresh]
    for c in winsorize_cols:
        low = float(np.quantile(kept[c].values, q_low))
        high = float(np.quantile(kept[c].values, q_high))
        winsor_caps[c] = {"low": low, "high": high}
        kept[c] = np.clip(kept[c].values, low, high)
    print(f"Winsorized {len(winsorize_cols)} skewed columns")

    corr = kept.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop_corr = set()

    max_iterations = len(kept.columns) // 2 
    iteration = 0

    while iteration < max_iterations:
        max_val = np.nanmax(upper.values)
        if max_val <= corr_thresh:
            break

        max_idx = np.unravel_index(np.nanargmax(upper.values), upper.shape)
        i, j = max_idx
        c1 = upper.index[i]
        c2 = upper.columns[j]

        c1_is_core = c1 in CORE_DESCRIPTORS
        c2_is_core = c2 in CORE_DESCRIPTORS

        if c1_is_core and not c2_is_core:
            drop = c2
        elif c2_is_core and not c1_is_core:
            drop = c1
        else:
            mean_corr = corr.loc[[c1, c2], :].mean()
            drop = c1 if mean_corr[c1] >= mean_corr[c2] else c2

        if len(kept.columns) - len(to_drop_corr) - 1 < min_features:
            print(f"Stopping correlation filter to preserve minimum {min_features} features")
            break

        to_drop_corr.add(drop)
        upper.loc[drop, :] = np.nan
        upper.loc[:, drop] = np.nan
        iteration += 1

    if to_drop_corr:
        kept = kept.drop(columns=list(to_drop_corr))
        print(f"After correlation filter: {len(kept.columns)} (-{len(to_drop_corr)})")

    if len(kept.columns) == 0:
        print("WARNING: All features were filtered out! Keeping core descriptors...")
        core_available = [c for c in CORE_DESCRIPTORS if c in imputed.columns]
        if core_available:
            kept = imputed[core_available].copy()
        else:
            non_empty = imputed.dropna(axis=1, how='all').columns[:10]
            kept = imputed[non_empty].copy()
        print(f"Emergency fallback: kept {len(kept.columns)} descriptors")

    scalers = {}
    scaler_types = {}
    for c in kept.columns:
        series = kept[c].astype(float)
        skew = float(series.skew()) if len(series.dropna()) > 1 else 0.0
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
    indicator_cols = [c for c in kept.columns if missing_rates.get(c, 0.0) >= indicator_thresh]

    imputer_fname = os.path.join(DEFAULT_SCALER_DIR, f"desc_imputer_{imputer_type}.pkl")
    scalers_fname = os.path.join(DEFAULT_SCALER_DIR, "desc_scalers.pkl")
    joblib.dump(imputer, imputer_fname)
    joblib.dump(scalers, scalers_fname)

    meta = {
        "all_input_cols": all_input_cols,
        "kept_cols": list(kept.columns),
        "original_kept_cols": variance_kept,
        "dropped_zero_cols": drop_zero,
        "dropped_variance_cols": variance_dropped,
        "dropped_corr_cols": sorted(list(to_drop_corr)),
        "winsor_caps": winsor_caps,
        "indicator_cols": indicator_cols,
        "imputer": {"type": imputer_type, "path": imputer_fname},
        "scalers": {"path": scalers_fname, "per_column": scaler_types},
        "missing_thresh": float(missing_thresh),
        "skew_thresh": float(skew_thresh),
        "indicator_thresh": float(indicator_thresh),
        "zero_ratio_thresh": float(zero_ratio_thresh),
        "var_thresh": float(var_thresh),
        "corr_thresh": float(corr_thresh),
        "winsor_q": [float(q_low), float(q_high)],
        "min_features": min_features
    }
    meta["descriptor_dim"] = len(meta["kept_cols"]) + len(meta["indicator_cols"])

    print(f"Final: {meta['descriptor_dim']} total features ({len(meta['kept_cols'])} descriptors + {len(meta['indicator_cols'])} indicators)")

    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)

    return meta

def transform_descriptors(df, meta=None):
    """
    Align df, impute, apply winsorization, scale, append missing indicators.
    """
    if meta is None:
        with open(META_PATH, "r") as fh:
            meta = json.load(fh)

    imputer = joblib.load(meta["imputer"]["path"])
    scalers = joblib.load(meta["scalers"]["path"])
    kept = meta["original_kept_cols"]
    kept_now = meta["kept_cols"]
    indicator_cols = meta.get("indicator_cols", [])
    winsor_caps = meta.get("winsor_caps", {})

    df2 = df.copy()

    for c in meta["all_input_cols"]:
        if c not in df2.columns:
            df2[c] = np.nan

    df2 = df2[kept]

    ind_arrays = []
    if indicator_cols:
        for c in indicator_cols:
            if c in df2.columns:
                ind_arrays.append(df2[c].isnull().astype(int).values.reshape(-1))

    imputed = pd.DataFrame(imputer.transform(df2), columns=kept, index=df2.index)

    for c, caps in winsor_caps.items():
        if c in imputed.columns:
            low, high = caps["low"], caps["high"]
            imputed[c] = np.clip(imputed[c].values, low, high)

    for c in meta.get("dropped_corr_cols", []):
        if c in imputed.columns:
            imputed = imputed.drop(columns=c)

    out_cols = []
    for c in meta["kept_cols"]:
        if c in scalers:
            sc = scalers[c]
            col_scaled = sc.transform(imputed[[c]]).reshape(-1)
            out_cols.append(col_scaled)

    if not out_cols:
        print("WARNING: No features to scale! Using zeros...")
        X_numeric = np.zeros((len(df2), 1))
    else:
        X_numeric = np.vstack(out_cols).T

    if ind_arrays and len(ind_arrays) > 0:
        X_ind = np.vstack(ind_arrays).T
        X = np.concatenate([X_numeric, X_ind], axis=1)
    else:
        X = X_numeric

    return X

def load_featurization_meta(path=META_PATH):
    with open(path, "r") as fh:
        return json.load(fh)

def compute_and_transform(smiles_series, fit_meta=None, use_vsa=True, use_3d=True):
    raw = compute_descriptor_df_enhanced(smiles_series, use_vsa=use_vsa, use_3d=use_3d)
    meta = fit_meta if fit_meta is not None else load_featurization_meta()
    return transform_descriptors(raw, meta)

def build_processed_splits(train_df, val_df, test_df, fit_meta=None, use_vsa=True, use_3d=True, fit_meta_on_train=True):
    if fit_meta is None and fit_meta_on_train:
        unique_smiles = pd.Index(sorted(train_df['Drug'].dropna().unique()))
        raw_tr_all = compute_descriptor_df_enhanced(unique_smiles, use_vsa=use_vsa, use_3d=use_3d)
        meta = fit_descriptor_pipeline(raw_tr_all)
    elif fit_meta is None:
        meta = load_featurization_meta()
    else:
        meta = fit_meta

    descriptor_dim = meta["descriptor_dim"]
    processed = {}
    targ_scalers = {}

    def get_atom_features(a):
        return [a.GetAtomicNum(), a.GetDegree(), int(a.GetHybridization()),
                int(a.GetIsAromatic()), a.GetFormalCharge(), int(a.IsInRing())]

    def get_bond_features(b):
        return [b.GetBondTypeAsDouble(), int(b.GetIsConjugated()),
                int(b.IsInRing()), int(b.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]

    def mol_graph_feats(series):
        A, B = [], []
        for smi in series:
            mol = Chem.MolFromSmiles(smi) if pd.notnull(smi) else None
            if mol:
                A.append([get_atom_features(a) for a in mol.GetAtoms()])
                B.append([get_bond_features(b) for b in mol.GetBonds()])
            else:
                A.append([])
                B.append([])
        return A, B

    tasks = sorted(train_df['task'].unique())
    for task in tasks:
        tr = train_df[train_df['task'] == task].reset_index(drop=True)
        vl = val_df[val_df['task'] == task].reset_index(drop=True)
        te = test_df[test_df['task'] == task].reset_index(drop=True)

        tr_X = compute_and_transform(tr['Drug'], fit_meta=meta, use_vsa=use_vsa, use_3d=use_3d)
        vl_X = compute_and_transform(vl['Drug'], fit_meta=meta, use_vsa=use_vsa, use_3d=use_3d)
        te_X = compute_and_transform(te['Drug'], fit_meta=meta, use_vsa=use_vsa, use_3d=use_3d)

        if len(tr) and tr.loc[0, 'task_type'] == 'regression':
            ts = StandardScaler().fit(tr[['Y']])
            targ_scalers[task] = ts
            tr_Y = ts.transform(tr[['Y']]).ravel()
            vl_Y = ts.transform(vl[['Y']]).ravel()
            te_Y = ts.transform(te[['Y']]).ravel()
        else:
            tr_Y = tr['Y'].values if len(tr) else np.array([])
            vl_Y = vl['Y'].values if len(vl) else np.array([])
            te_Y = te['Y'].values if len(te) else np.array([])

        tr_A, tr_B = mol_graph_feats(tr['Drug'])
        vl_A, vl_B = mol_graph_feats(vl['Drug'])
        te_A, te_B = mol_graph_feats(te['Drug'])

        processed[task] = {
            'train': {'desc': tr_X, 'Y': tr_Y, 'atoms': tr_A, 'bonds': tr_B, 'smiles': tr['Drug'].values},
            'val':   {'desc': vl_X, 'Y': vl_Y, 'atoms': vl_A, 'bonds': vl_B, 'smiles': vl['Drug'].values},
            'test':  {'desc': te_X, 'Y': te_Y, 'atoms': te_A, 'bonds': te_B, 'smiles': te['Drug'].values},
        }

    return processed, targ_scalers, meta, descriptor_dim


