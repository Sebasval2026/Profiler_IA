"""Validación post-corte: evalúa el híbrido v2 sobre decisiones reales
posteriores a una fecha (que los modelos no vieron), leyéndolas de la base.

    PG_URL=postgresql://... python3 entrenamiento/validar_fresco.py 2026-08-18

Sin PG_URL reutiliza los CSV ya exportados en datos/validacion/. Reconstruye
Experian/AgilData as-of desde migration (la MV los tiene rotos; los nulos de
migration son TEXTO: toda coerción va con errors='coerce') y el historial por
lender desde la MV. Cada solicitud pasa por api.inferencia_v2.evaluar_v2 —
el mismo camino del servicio, training-serving skew incluido.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from api import inferencia, inferencia_v2  # noqa: E402

DIR = RAIZ / 'datos' / 'validacion'
MV = 'bi_analysis.mv_master_creditop_request_selected'
EX_CAMPOS = inferencia_v2.EX_CAMPOS


def exportar(pg_url, desde):
    DIR.mkdir(parents=True, exist_ok=True)
    frescos = ("SELECT DISTINCT ur_user_id FROM %s WHERE ur_created_at >= '%s' "
               "AND urs_name IN ('Autorizada','Negada')" % (MV, desde))
    ex_cols = ', '.join('e.ex_' + c for c in EX_CAMPOS)
    consultas = {
        'base': "SELECT ur_id, ur_user_id::text AS uid, ur_lender_id, l_name, urs_name, "
                "ur_created_at, ur_amount, ur_fee_number, a_id, a_allied_industry_id, "
                "a_allied_type_id, ur_credit_line_id, u_age, u_gender, u_created_at, "
                "mg_average_income, mg_average_income_reported, mg_continuity, "
                "mg_continuity_reported, mg_occupation FROM %s "
                "WHERE ur_created_at >= '%s' AND urs_name IN ('Autorizada','Negada')" % (MV, desde),
        'hist': "SELECT ur_user_id::text AS uid, ur_lender_id, urs_name, ur_created_at "
                "FROM %s WHERE urs_name IN ('Autorizada','Negada') AND ur_user_id IN (%s)" % (MV, frescos),
        'ex': 'SELECT e.ex_user_id AS uid, e.created_at, %s FROM migration."Risk_Central_Experian" e '
              'WHERE e.ex_user_id IN (SELECT ur_user_id::text FROM (%s) s)' % (ex_cols, frescos),
        'ad': 'SELECT a.ad_user_id AS uid, a.created_at, a.ad_age, a.ad_genre, '
              'a.ad_average_income, a.ad_continuity FROM migration."Risk_Central_AgilData" a '
              'WHERE a.ad_user_id IN (SELECT ur_user_id::text FROM (%s) s)' % frescos,
    }
    for nombre, q in consultas.items():
        destino = DIR / ('%s.csv' % nombre)
        subprocess.run(['psql', pg_url, '-c',
                        "\\copy (%s) TO '%s' CSV HEADER" % (q, destino)], check=True)
        print('  %s.csv listo' % nombre)


def cargar():
    base = pd.read_csv(DIR / 'base.csv', dtype={'uid': str})
    hist = pd.read_csv(DIR / 'hist.csv', dtype={'uid': str})
    ex = pd.read_csv(DIR / 'ex.csv', dtype={'uid': str}, low_memory=False)
    ad = pd.read_csv(DIR / 'ad.csv', dtype={'uid': str}, low_memory=False)
    for c in ex.columns:
        if c not in ('uid', 'created_at'):
            ex[c] = pd.to_numeric(ex[c], errors='coerce')
    for c in ('ad_age', 'ad_average_income', 'ad_continuity'):
        ad[c] = pd.to_numeric(ad[c], errors='coerce')
    ex['created_at'] = pd.to_datetime(ex['created_at'], errors='coerce')
    ad['created_at'] = pd.to_datetime(ad['created_at'], errors='coerce')
    hist['ur_created_at'] = pd.to_datetime(hist['ur_created_at'])
    base['ur_created_at'] = pd.to_datetime(base['ur_created_at'])
    return base, hist, ex, ad


def _v(x):
    return None if pd.isna(x) else float(x)


def _i(x):
    return int(x) if pd.notna(x) else None


def payload_item(r, ex_uid, ad_uid, hist_uid):
    def asof(por_uid):
        g = por_uid.get(r['uid'])
        if g is None:
            return None
        g = g[g['created_at'] < r['ur_created_at']]
        return g.iloc[-1] if len(g) else None

    item = {'solicitud': {'amount': float(r['ur_amount']),
                          'fee_number': _i(r['ur_fee_number']),
                          'fecha_solicitud': r['ur_created_at'].isoformat(),
                          'allied_id': _i(r['a_id']),
                          'allied_industry_id': _i(r['a_allied_industry_id']),
                          'allied_type_id': _i(r['a_allied_type_id']),
                          'credit_line_id': _i(r['ur_credit_line_id'])},
            'usuario': {'age': _v(r['u_age']),
                        'gender': r['u_gender'] if pd.notna(r['u_gender']) else None,
                        'user_created_at': str(r['u_created_at']) if pd.notna(r['u_created_at']) else None}}
    mg = {'average_income': _v(r['mg_average_income']),
          'average_income_reported': _v(r['mg_average_income_reported']),
          'continuity': _v(r['mg_continuity']),
          'continuity_reported': _v(r['mg_continuity_reported']),
          'occupation': r['mg_occupation'] if pd.notna(r['mg_occupation']) else None}
    if any(v is not None for v in mg.values()):
        item['mareigua'] = mg
    e = asof(ex_uid)
    if e is not None:
        item['experian'] = {c: _v(e['ex_' + c]) for c in EX_CAMPOS}
    a = asof(ad_uid)
    if a is not None:
        item['agildata'] = {'average_income': _v(a['ad_average_income']),
                            'continuity': _v(a['ad_continuity']),
                            'age': _v(a['ad_age']),
                            'genre': a['ad_genre'] if pd.notna(a['ad_genre']) else None}
    g = hist_uid.get(r['uid'])
    if g is not None:
        g = g[g['ur_created_at'] < r['ur_created_at']]  # < estricto
        if len(g):
            por = {str(int(lid)): {'aprobadas': int((gg['urs_name'] == 'Autorizada').sum()),
                                   'negadas': int((gg['urs_name'] == 'Negada').sum()),
                                   'fecha_ultima_decidida': gg['ur_created_at'].max().isoformat()}
                   for lid, gg in g.groupby('ur_lender_id')}
            item['historial'] = {'por_lender': por}
    return item


def metricas(g):
    from sklearn.metrics import roc_auc_score
    dec = g[g.banda.isin([0, 2])]
    alta, baja = g[g.banda == 2], g[g.banda == 0]
    try:
        auc = roc_auc_score(g.y, g.proba) if g.y.nunique() == 2 else np.nan
    except ValueError:
        auc = np.nan
    return pd.Series({
        'n': len(g), 'tasa_base': g.y.mean(),
        'cobertura': len(dec) / len(g),
        'acc_0y2': ((dec.banda == 2) == (dec.y == 1)).mean() if len(dec) else np.nan,
        'prec_alta': alta.y.mean() if len(alta) else np.nan, 'n_alta': len(alta),
        'prec_baja': 1 - baja.y.mean() if len(baja) else np.nan, 'n_baja': len(baja),
        'auc': auc})


def main():
    import os
    desde = sys.argv[1] if len(sys.argv) > 1 else None
    pg_url = os.environ.get('PG_URL')
    if pg_url and desde:
        print('Exportando desde la base (>= %s)...' % desde)
        exportar(pg_url, desde)
    elif not (DIR / 'base.csv').exists():
        sys.exit('Sin PG_URL ni CSVs en %s. Uso: PG_URL=... %s <fecha_desde>'
                 % (DIR, sys.argv[0]))
    base, hist, ex, ad = cargar()
    ex_uid = {k: g.sort_values('created_at') for k, g in ex.groupby('uid')}
    ad_uid = {k: g.sort_values('created_at') for k, g in ad.groupby('uid')}
    hist_uid = {k: g for k, g in hist.groupby('uid')}

    A1 = inferencia.cargar_artefactos()
    A2 = inferencia_v2.cargar_artefactos_v2()
    filas = []
    for _, r in base.iterrows():
        res = inferencia_v2.evaluar_v2(
            A2, A1, {'lenders': [int(r['ur_lender_id'])],
                     'solicitudes': [payload_item(r, ex_uid, ad_uid, hist_uid)]})
        x = res['model_results'][0]
        filas.append({'ur_id': r['ur_id'], 'lender_id': int(r['ur_lender_id']),
                      'l_name': r['l_name'], 'modelo': x['model_name'],
                      'banda': x['prediction']['banda'],
                      'proba': x['prediction']['proba_autorizada'][0],
                      'y': int(r['urs_name'] == 'Autorizada')})
    d = pd.DataFrame(filas)
    d.to_csv(DIR / 'resultados.csv', index=False)

    print('\n=== GLOBAL (%d solicitudes post-corte) ===' % len(d))
    print(metricas(d).round(3).to_string())
    print('\n=== POR MODELO ===')
    print(d.groupby('modelo').apply(metricas, include_groups=False).round(3).to_string())
    print('\n=== POR LENDER (n>=20) ===')
    t = d.groupby(['lender_id', 'l_name']).apply(metricas, include_groups=False)
    print(t[t.n >= 20].round(3).to_string())
    print('\nDetalle por solicitud: %s' % (DIR / 'resultados.csv'))


if __name__ == '__main__':
    main()
