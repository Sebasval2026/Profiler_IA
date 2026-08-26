"""
Inferencia v2 — híbrido.

Modelos de boosting por lender (informe v2, 2026-08-18: CatBoost/LightGBM/
XGBoost, umbrales t_bajo/t_alto por lender, validación out-of-time) servidos
con la arquitectura stateless de v1. Diferencias deliberadas con el contrato
v2 puro:

  - fee_number es OPCIONAL: si es null se barren los planes (max-plan, v1) —
    el perfilador corre antes de que el usuario elija cuotas.
  - historial (bloque v1, por_lender exhaustivo) es entrada opcional: lo usa
    el modelo v1 de Meddipay, que v2 no cubre.
  - Meddipay (39) se sirve con el modelo v1 (logit/RF + isotónica); su banda
    alta no certificó, así que va como solo_descarte (clase 2 = revision).

Siempre se emiten las 3 bandas; el guardarraíl es modo_uso, no la supresión.
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import inferencia as v1

VERSION_V2 = 'v2.0-hibrido'
DIR_V2 = v1.DIR_MODELOS / 'v2'
DIAS_RECALIBRACION = 45          # calibrated_at más viejo -> warning
LEAKAGE = ('initial_fee', 'final_amount', 'amount_available', 'payment_amount')

MEDDIPAY = 39
MAPEO = {6: 'Addi_V2', 9: 'Sistecredito_V2', 5: 'BancoBogota_V2',
         19: 'Brilla_V2', 36: 'Davivienda_V2'}
GENERAL = 'General_V2'
# ponytail: PayJoy rechazado por el auditor v2 (leakage, AUC 0.56) y su
# lender_id no está confirmado en ninguna fuente local; agregarlo aquí
# cuando se confirme para responder sin_score en vez de caer al General.
SIN_SCORE = set()

MODO_USO = {'Addi_V2': 'decision', 'Sistecredito_V2': 'decision',
            'General_V2': 'decision', 'BancoBogota_V2': 'solo_descarte',
            'Davivienda_V2': 'solo_descarte', 'Brilla_V2': 'ordenamiento'}
CLASES = {'0': 'baja', '1': 'media', '2': 'alta'}
CLASES_DESCARTE = {'0': 'baja', '1': 'media', '2': 'revision'}

EX_CAMPOS = ['score', 'total_debt_balance', 'monthly_payment',
             'current_disputes', 'current_negative_credits',
             'negative_historical_last_12_months', 'consulted_last_6_months',
             'savings_quantity', 'savings_active', 'savings_seized',
             'cc_quantity', 'cc_active', 'cc_active_delinquent',
             'cc_initial_value_average', 'cc_initial_value', 'cc_debt_balance',
             'cc_available_balance', 'cc_monthly_payment', 'cc_vector_payed',
             'cc_vector_overdue', 'liabilities_quantity', 'liabilities_active',
             'liabilities_active_delinquent',
             'liabilities_initial_value_average', 'liabilities_initial_value',
             'liabilities_debt_balance', 'liabilities_monthly_payment']


# ---------------------------------------------------------------- artefactos

def _cats_validas(pipe, fam, feats_cat):
    """Categorías vistas en entrenamiento; valores fuera de la lista van a NaN
    (xgboost revienta con categorías nuevas; lightgbm las codificaría mal).
    CatBoost acepta strings arbitrarios: sin restricción."""
    if fam == 'lightgbm':
        return dict(zip(feats_cat, pipe.booster_.pandas_categorical))
    if fam == 'xgboost':
        arrow = pipe.get_booster().get_categories(export_to_arrow=True).to_arrow()
        return {k: v.to_pylist() for k, v in arrow if v is not None}
    return None


def cargar_artefactos_v2():
    A = {}
    if not DIR_V2.exists():
        return A
    for d in sorted(DIR_V2.iterdir()):
        if not d.is_dir():
            continue
        try:
            nombre = next(d.glob('model_*.joblib')).stem.replace('model_', '')
        except StopIteration:
            continue
        pipe = joblib.load(d / ('model_%s.joblib' % nombre))
        meta = json.load(open(d / ('metadata_%s.json' % nombre)))
        fam = meta['config']['model']
        A[nombre] = {
            'pipe': pipe, 'fam': fam, 'meta': meta,
            'bandas': json.load(open(d / ('bandas_config_%s.json' % nombre))),
            'cols': json.load(open(d / ('columnas_entrada_%s.json' % nombre))),
            'cats': _cats_validas(pipe, fam, meta['features_cat']),
        }
    return A


# ---------------------------------------------------------------- derivación

def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _primero(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _ratio(a, b):
    return a / b if (a is not None and b is not None and b != 0) else None


def derivar(s):
    """Item de solicitud (dict del payload v2) -> fila de features + warnings.
    Toda la tabla de derivación del contrato v2 vive acá."""
    w = []
    sol, usu = s['solicitud'], s['usuario']
    mar = s.get('mareigua') or {}
    agi = s.get('agildata') or {}
    ex = s.get('experian') or {}

    for campo in LEAKAGE:
        if sol.get(campo) is not None:
            w.append('campo_leakage_ignorado: %s' % campo)

    fs = datetime.fromisoformat(sol['fecha_solicitud'])

    edad = None
    if usu.get('date_of_birth'):
        dob = datetime.fromisoformat(usu['date_of_birth']).date()
        edad = (fs.date() - dob).days / 365.25
    edad = _primero(edad, _num(usu.get('age')), _num(agi.get('age')))
    if edad is not None and not (18 <= edad <= 95):
        edad = None

    g = _primero(usu.get('gender'), agi.get('genre'))
    genero = str(g).strip()[:1].upper() if g else 'N'
    if genero not in ('M', 'F'):
        genero = 'N'

    tenure = 0
    if usu.get('user_created_at'):
        uc = datetime.fromisoformat(usu['user_created_at'])
        tenure = max(0, (fs.date() - uc.date()).days)

    income = _primero(_num(mar.get('average_income')),
                      _num(agi.get('average_income')),
                      _num(mar.get('average_income_reported')))
    if income is None:
        w.append('sin_bureau_de_ingresos')
    continuidad = _primero(_num(mar.get('continuity')),
                           _num(agi.get('continuity')),
                           _num(mar.get('continuity_reported')))
    ocupacion = (mar.get('occupation') or '').strip().lower() or 'desconocida'

    monto = float(sol['amount'])
    exf = {('ex_' + c): _num(ex.get(c)) for c in EX_CAMPOS}
    cc_debt, cc_avail = exf['ex_cc_debt_balance'], exf['ex_cc_available_balance']
    cc_total = (cc_debt + cc_avail) if (cc_debt is not None and cc_avail is not None) else None

    fila = {
        'u_age': edad, 'gender': genero, 'user_tenure_days': float(tenure),
        'amount': monto, 'ur_fee_number': _num(sol.get('fee_number')),
        'income_best': income, 'continuity_best': continuidad,
        'occupation': ocupacion,
        'amount_to_income': _ratio(monto, income),
        'debt_to_income': _ratio(exf['ex_total_debt_balance'], income),
        'payment_to_income': _ratio(exf['ex_monthly_payment'], income),
        'cc_utilization': _ratio(cc_debt, cc_total),
        'allied_industry': str(sol['allied_industry_id']) if sol.get('allied_industry_id') is not None else None,
        'allied_type': str(sol['allied_type_id']) if sol.get('allied_type_id') is not None else None,
        'credit_line': str(sol['credit_line_id']) if sol.get('credit_line_id') is not None else None,
        'has_experian': float(any(v is not None for v in exf.values())),
        'has_agildata': float(any(_num(agi.get(k)) is not None for k in ('average_income', 'continuity', 'age'))),
        'has_mareigua': float(any(_num(mar.get(k)) is not None for k in
                                  ('average_income', 'average_income_reported', 'continuity', 'continuity_reported'))),
        # contexto para las rutas v1 (no son features v2):
        '_fs': fs, '_historial': s.get('historial'),
        '_allied_id': sol.get('allied_id'),
        '_industry_id': sol.get('allied_industry_id'),
    }
    fila.update(exf)
    return fila, w


# ---------------------------------------------------------------- predicción

def _df_para(m, fila):
    df = pd.DataFrame([{c: fila.get(c) for c in m['cols']}])
    cats = set(m['meta']['features_cat'])
    for c in m['cols']:
        if c in cats:
            v = fila.get(c)
            if m['fam'] == 'catboost':
                # entrenado bajo pandas 2: astype(str) volvía NaN el string 'nan'
                df[c] = pd.Series([str(v) if v is not None else 'nan'], dtype=object)
            else:
                validas = (m['cats'] or {}).get(c, [])
                df[c] = pd.Categorical([v if v in validas else None], categories=validas)
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _proba(m, fila):
    return float(m['pipe'].predict_proba(_df_para(m, fila)[m['cols']])[:, 1][0])


def _proba_maxplan(m, fila):
    """Sin plazo elegido: máximo sobre planes (semántica de enrutamiento v1)."""
    if fila.get('ur_fee_number') is not None or 'ur_fee_number' not in m['cols']:
        return _proba(m, fila), None
    mejor_p, mejor_plan = -1.0, None
    for plan in v1.PLANES:
        p = _proba(m, dict(fila, ur_fee_number=float(plan)))
        if p > mejor_p:
            mejor_p, mejor_plan = p, plan
    return mejor_p, mejor_plan


def _banda_v2(p, bandas):
    return 0 if p <= bandas['t_bajo'] else (2 if p >= bandas['t_alto'] else 1)


def _banda_v1(p):
    return 2 if p >= v1.UA else (0 if p < v1.UB else 1)


def _resultado(lender_id, model_name, bandas_l, probas, clases, umbrales,
               modo_uso, version, calibrated_at=None, mejor_plan=None):
    pred = {'banda': bandas_l[0], 'banda_probabilidad': bandas_l,
            'proba_autorizada': [round(p, 6) for p in probas],
            'clases': clases, 'umbrales': umbrales, 'modo_uso': modo_uso}
    if mejor_plan is not None:
        pred['mejor_plan'] = mejor_plan
    return {'lender_id': lender_id, 'model_name': model_name,
            'model_version': version, 'calibrated_at': calibrated_at,
            'prediction': pred}


def _tasa_fallback(A1, lender_id, fila):
    """Sin modelo aplicable: tasa observada (lender, comercio) — v1."""
    aid = fila.get('_allied_id')
    n, t = A1['tasas'].get((lender_id, aid), (0, None)) if aid is not None else (0, None)
    if n < v1.MIN_N_TASA:
        n, t = A1['tasas'].get((lender_id, -1), (0, None))
    if t is None:
        return 1, None, 'sin_historia'
    return _banda_v1(t), t, 'tasa_observada'


def _evaluar_meddipay(A1, filas):
    """Meddipay con el modelo v1 (v2 no lo cubre). Banda alta no certificada
    -> solo_descarte: clase 2 se rotula 'revision'."""
    bandas_l, probas, planes, servido = [], [], [], set()
    for fila in filas:
        score = v1.sanear_score(fila.get('ex_score'))
        modelo = A1.get('meddipay_central')
        if score is not None and modelo is not None:
            hl, dias_ult = v1.clasificar_historial(fila['_historial'], MEDDIPAY, fila['_fs'])
            df = pd.DataFrame([{'a_allied_industry_id': fila['_industry_id'],
                                'monto': fila['amount'], 'edad': fila['u_age'],
                                'antig_ctop': fila['user_tenure_days'],
                                'dias_ult': dias_ult, 'hist_lender': hl,
                                'ex_score': score}])
            if fila.get('ur_fee_number') is not None:
                df['cuotas'] = fila['ur_fee_number']
                p, plan = v1.predecir(modelo, df), None
            else:
                p, plan = v1.predecir_max(modelo, df)
            bandas_l.append(_banda_v1(p)); probas.append(p); planes.append(plan)
            servido.add('Meddipay_V1')
        else:
            b, t, via = _tasa_fallback(A1, MEDDIPAY, fila)
            bandas_l.append(b); probas.append(t if t is not None else 0.5)
            planes.append(None); servido.add(via)
    nombre = 'Meddipay_V1' if 'Meddipay_V1' in servido else servido.pop()
    modo = 'solo_descarte' if nombre == 'Meddipay_V1' else 'ordenamiento'
    return _resultado(MEDDIPAY, nombre, bandas_l, probas, CLASES_DESCARTE,
                      {'t_bajo': v1.UB, 't_alto': v1.UA}, modo, v1.VERSION,
                      mejor_plan=planes[0])


def evaluar_v2(A2, A1, payload):
    """Payload v2 (dict ya validado) -> respuesta del contrato v2."""
    warnings, filas = [], []
    for s in payload['solicitudes']:
        fila, w = derivar(s)
        filas.append(fila); warnings.extend(w)
    fecha_ref = filas[0]['_fs'].date()

    results = []
    for lid in payload['lenders']:
        if lid in SIN_SCORE:
            results.append({'lender_id': lid, 'model_name': 'sin_score',
                            'model_version': VERSION_V2, 'calibrated_at': None,
                            'prediction': {'banda': None, 'modo_uso': 'sin_score'}})
            continue
        if lid == MEDDIPAY:
            results.append(_evaluar_meddipay(A1, filas))
            continue
        nombre = MAPEO.get(lid, GENERAL)
        m = A2.get(nombre)
        if m is None:
            warnings.append('modelo_no_disponible: %s' % nombre)
            bandas_l, probas = [], []
            for fila in filas:
                b, t, _ = _tasa_fallback(A1, lid, fila)
                bandas_l.append(b); probas.append(t if t is not None else 0.5)
            results.append(_resultado(lid, 'tasa_observada', bandas_l, probas,
                                      CLASES, {'t_bajo': v1.UB, 't_alto': v1.UA},
                                      'ordenamiento', v1.VERSION))
            continue
        bandas_l, probas, planes = [], [], []
        for fila in filas:
            p, plan = _proba_maxplan(m, fila)
            bandas_l.append(_banda_v2(p, m['bandas'])); probas.append(p)
            planes.append(plan)
        modo = MODO_USO[nombre]
        calib = m['bandas'].get('calibrated_at')
        if calib and (fecha_ref - datetime.fromisoformat(calib).date()).days > DIAS_RECALIBRACION:
            warnings.append('umbrales_desactualizados: %s' % nombre)
        results.append(_resultado(
            lid, nombre, bandas_l, probas,
            CLASES_DESCARTE if modo == 'solo_descarte' else CLASES,
            {'t_bajo': round(m['bandas']['t_bajo'], 6),
             't_alto': round(m['bandas']['t_alto'], 6)},
            modo, '2.0', calibrated_at=calib, mejor_plan=planes[0]))

    eco = {k: (round(fila[k], 4) if isinstance(fila[k], float) else fila[k])
           for fila in filas[:1] for k in
           ('income_best', 'continuity_best', 'has_experian', 'has_agildata',
            'has_mareigua', 'amount_to_income', 'debt_to_income',
            'payment_to_income', 'cc_utilization', 'u_age',
            'user_tenure_days', 'gender', 'occupation')}
    return {'model_results': results, 'features_derivadas': eco,
            'warnings': sorted(set(warnings))}
