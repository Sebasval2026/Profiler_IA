"""
Core de inferencia del perfilador. Stateless: todo entra por el payload.

Semantica de bandas (v1.4):
  - Segmentos con modelo: barrido sobre PLANES de cuotas (el cliente aun no
    eligio plan). alta = existe al menos un plan con probabilidad alta;
    baja = ningun plan alcanza. Solo se emiten las bandas CERTIFICADAS del
    segmento (gates en dos cortes temporales); el resto cae a media.
  - Segmentos sin modelo: la tasa observada (lender, comercio) pasa por los
    mismos umbrales. Es una frecuencia: esta calibrada por construccion.
  - media = el sistema no se pronuncia. Absorbe tambien el caso sin datos.

El historial llega del cliente como conteos primitivos POR TODOS los lenders
donde el usuario tiene solicitudes decididas (contrato: por_lender exhaustivo).
La clasificacion a 6 niveles vive aca, en un solo lugar.
"""  # ponytail: sin modo sombra — version basica; reintroducir si se necesita telemetria
import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

VERSION = 'v1.4-contrato-min'
UA, UB = 0.65, 0.35            # umbrales de banda, compartidos entre lenders
PLANES = [6, 12, 18, 24, 36]   # planes de cuotas barridos por el max-plan
MIN_N_TASA = 50                # bajo esto, la tasa por comercio cae al global
MAPA_BANDA = {'baja': 0, 'media': 1, 'alta': 2}

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / 'modelos'

# Bandas certificadas por segmento (salida de entrenamiento/certificar.py,
# gates: n>=30, precision>=70, uplift>=5pp, en AMBOS cortes temporales).
CERTIFICADAS = {
    'addi_central': ['alta', 'baja'],
    'addi_thin': ['baja'],
    'meddipay_central': ['baja'],
    'generalista_supay': ['baja'],
}


def cargar_artefactos():
    A = {}
    for n in ('addi_central', 'addi_thin', 'meddipay_central', 'generalista'):
        p = DIR_MODELOS / ('modelo_%s.joblib' % n)
        A[n] = joblib.load(p) if p.exists() else None
    A['politica'] = json.load(open(DIR_MODELOS / 'politica.json'))
    tasas = {}
    for r in csv.DictReader(open(DIR_MODELOS / 'tasas.csv')):
        tasas[(int(r['lid']), int(r['aid']))] = (int(r['n']), float(r['tasa']))
    A['tasas'] = tasas
    return A


def clasificar_historial(hist, lender_id, fecha_solicitud):
    """Conteos primitivos -> nivel de historial. UNICA definicion del sistema.

    Debe coincidir 1:1 con la window function de datos/extraccion/features.sql.
    La prueba de tests/test_consistencia.py verifica esa coincidencia sobre
    1.000 solicitudes historicas antes de cada release.
    """
    por = (hist or {}).get('por_lender', {})
    e = por.get(str(lender_id), {}) or {}
    ap = int(e.get('aprobadas') or 0)
    ng = int(e.get('negadas') or 0)
    # por_lender es exhaustivo (contrato): la suma es el total exacto
    n_total = sum(int(x.get('aprobadas') or 0) + int(x.get('negadas') or 0)
                  for x in por.values())
    if n_total == 0:
        hl = 'primera_vez'
    elif ap == 0 and ng == 0:
        hl = 'nunca_aqui'
    elif ap > 0 and ng == 0:
        hl = 'solo_aprobado_aqui'
    elif ap > 0 and ng > 0:
        hl = 'mixto_aqui'
    elif ng == 1:
        hl = 'negado_1_aqui'
    else:
        hl = 'negado_2mas_aqui'
    fechas = [x.get('fecha_ultima_decidida') for x in por.values()
              if x.get('fecha_ultima_decidida')]
    dias_ult = None
    if fechas:
        dias_ult = (fecha_solicitud - datetime.fromisoformat(max(fechas))).days
    return hl, dias_ult


def sanear_score(v):
    """0 y 1 son codigos de Datacredito, no scores."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 1 else None


def features_para(payload, lender_id):
    s, u = payload['solicitud'], payload['usuario']
    fs = datetime.fromisoformat(s['fecha_solicitud'])
    uc = datetime.fromisoformat(u['user_created_at'])
    hl, dias_ult = clasificar_historial(payload.get('historial'), lender_id, fs)
    score = sanear_score((payload.get('experian') or {}).get('score'))
    fila = {
        'a_allied_industry_id': s['allied_industry_id'],
        'monto': float(s['amount']),
        'edad': float(u['age']),
        'antig_ctop': (fs.date() - uc.date()).days,
        'dias_ult': dias_ult,
        'hist_lender': hl,
        'ex_score': score,
        'ur_lender_id': lender_id,
    }
    return pd.DataFrame([fila]), score is not None


def predecir(modelo, df):
    p = modelo['pipe'].predict_proba(df[modelo['feats']])[:, 1]
    return float(modelo['iso'].predict(p)[0])


def predecir_max(modelo, df):
    """Maximo sobre planes: el usuario aun no eligio cuotas."""
    mejor_p, mejor_plan = -1.0, None
    for plan in PLANES:
        d2 = df.copy()
        d2['cuotas'] = float(plan)
        p = predecir(modelo, d2)
        if p > mejor_p:
            mejor_p, mejor_plan = p, plan
    return mejor_p, mejor_plan


def banda(p, certificadas):
    if p >= UA and 'alta' in certificadas:
        return 'alta'
    if p < UB and 'baja' in certificadas:
        return 'baja'
    return 'media'


def evaluar_lender(A, payload, lender_id):
    aid = int(payload['solicitud']['allied_id'])
    r = {'lender_id': lender_id, 'modelo_version': VERSION}
    df, con_score = features_para(payload, lender_id)
    if lender_id == 6:
        seg = 'addi_central' if con_score else 'addi_thin'
        p, plan = predecir_max(A[seg], df)
        r.update(estado=banda(p, CERTIFICADAS[seg]), probabilidad=round(p, 3),
                 mejor_plan=plan, servido_por='modelo_propio:%s:maxplan' % seg)
        return r
    if lender_id == 39 and con_score:
        p, plan = predecir_max(A['meddipay_central'], df)
        r.update(estado=banda(p, CERTIFICADAS['meddipay_central']),
                 probabilidad=round(p, 3), mejor_plan=plan,
                 servido_por='modelo_propio:meddipay_central:maxplan')
        return r
    if lender_id == 11:
        p = predecir(A['generalista'], df)
        r.update(estado=banda(p, CERTIFICADAS['generalista_supay']),
                 probabilidad=round(p, 3), servido_por='generalista')
        return r
    # Sin modelo certificado: la tasa observada pasa por los mismos umbrales.
    n, t = A['tasas'].get((lender_id, aid), (0, None))
    if n < MIN_N_TASA:
        n, t = A['tasas'].get((lender_id, -1), (0, None))
    if t is None:
        r.update(estado='media', servido_por='sin_historia')
        return r
    r.update(estado=banda(t, ['alta', 'baja']), probabilidad=round(t, 3),
             servido_por='tasa_observada', soporte_n=n)
    return r


def evaluar(A, payload):
    """Payload (dict ya validado) -> contrato minimo."""
    interno = [evaluar_lender(A, payload, l) for l in payload['lenders']]
    return {'request_id': payload['request_id'],
            'model_results': [{'lender_id': x['lender_id'],
                               'prediction': {'approval_band': MAPA_BANDA.get(x['estado'], 1)}}
                              for x in interno]}
