"""Regenera los 4 .joblib de produccion en modelos/.

Entrena con train<2026-04-01 y calibra (isotonica) con [2026-04-01, 2026-05-15),
el mismo esquema con que se certificaron las bandas. Los CSV van en datos/.

    python3 entrenamiento/entrenar.py
"""
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore')
RAIZ = Path(__file__).resolve().parent.parent
DATOS, MODELOS = RAIZ / 'datos', RAIZ / 'modelos'
CATS = ['a_allied_industry_id', 'hist_lender', 'ur_lender_id']
VERSION = 'v1.4'


def prep(feats, modo):
    cats = [c for c in feats if c in CATS]
    nums = [c for c in feats if c not in CATS]
    tr = []
    if cats:
        if modo == 'arbol':
            tr.append(('c', OrdinalEncoder(handle_unknown='use_encoded_value',
                      unknown_value=-1, encoded_missing_value=-2), cats))
        else:
            tr.append(('c', OneHotEncoder(handle_unknown='infrequent_if_exist',
                      min_frequency=30, sparse_output=False), cats))
    if modo == 'arbol':
        tr.append(('n', SimpleImputer(strategy='median', add_indicator=True), nums))
    else:
        tr.append(('n', Pipeline([
            ('i', SimpleImputer(strategy='median', add_indicator=True)),
            ('s', StandardScaler())]), nums))
    return ColumnTransformer(tr)


def arma(d, feats, modo, mk, nombre):
    t = pd.to_datetime(d['ur_created_at'])
    TR = d[t < '2026-04-01']
    CA = d[(t >= '2026-04-01') & (t < '2026-05-15')]
    pipe = Pipeline([('p', prep(feats, modo)), ('m', mk())])
    pipe.fit(TR[feats], TR['y'])
    iso = IsotonicRegression(out_of_bounds='clip').fit(
        pipe.predict_proba(CA[feats])[:, 1], CA['y'])
    joblib.dump({'pipe': pipe, 'iso': iso, 'feats': feats,
                 'umbrales': [0.65, 0.35], 'version': VERSION},
                MODELOS / ('modelo_%s.joblib' % nombre))
    print('  modelo_%s.joblib (train=%d, calib=%d)' % (nombre, len(TR), len(CA)))


def main():
    LOG = lambda: LogisticRegression(max_iter=3000, C=0.5)
    RF = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=15,
                                        n_jobs=-1, random_state=42)
    d6 = pd.read_csv(DATOS / 'features_6.csv', low_memory=False)
    d39 = pd.read_csv(DATOS / 'features_39.csv', low_memory=False)
    dg = pd.read_csv(DATOS / 'features_gen.csv', low_memory=False)
    # 'cuotas' SI entra al entrenamiento: en inferencia se barre (max-plan)
    B = ['a_allied_industry_id', 'monto', 'cuotas', 'edad']
    arma(d6[d6.ex_score.notna()], B + ['hist_lender', 'ex_score'], 'lineal', LOG, 'addi_central')
    arma(d6[d6.ex_score.isna()], B + ['antig_ctop', 'dias_ult', 'hist_lender'], 'lineal', LOG, 'addi_thin')
    arma(d39[d39.ex_score.notna()], B + ['hist_lender', 'ex_score'], 'arbol', RF, 'meddipay_central')
    Bs = ['a_allied_industry_id', 'monto', 'edad']   # generalista: sin cuotas
    arma(dg, ['ur_lender_id'] + Bs + ['antig_ctop', 'dias_ult', 'hist_lender'], 'arbol', RF, 'generalista')


if __name__ == '__main__':
    main()
