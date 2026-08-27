"""Tests del contrato v2 hibrido. Requieren los artefactos de modelos/v2/."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import inferencia, inferencia_v2  # noqa: E402

pytestmark = pytest.mark.skipif(not inferencia_v2.DIR_V2.exists(),
                                reason='requiere artefactos en modelos/v2/')

PAYLOAD = {
    'lenders': [6, 9, 5, 39, 999],   # Addi, Sistecredito, BdB, Meddipay, no mapeado
    'solicitudes': [{
        'solicitud': {'amount': 4500000, 'fee_number': 24,
                      'fecha_solicitud': '2026-08-18T10:30:00',
                      'allied_id': 26, 'allied_industry_id': 3,
                      'allied_type_id': 1, 'credit_line_id': 1,
                      'initial_fee': 200000},   # leakage: debe ignorarse y advertir
        'usuario': {'date_of_birth': '1991-04-12', 'gender': 'F',
                    'user_created_at': '2025-11-02T09:15:00'},
        'mareigua': {'average_income': 2800000, 'continuity': 14,
                     'occupation': 'Empleado'},
        'agildata': {'average_income': 2650000, 'continuity': 11, 'age': 35},
        'experian': {'score': 742, 'total_debt_balance': 5300000,
                     'monthly_payment': 480000, 'cc_debt_balance': 700000,
                     'cc_available_balance': 1300000},
        'historial': {'por_lender': {
            '39': {'aprobadas': 1, 'negadas': 0,
                   'fecha_ultima_decidida': '2026-07-04T16:20:00'}}},
    }],
}


@pytest.fixture(scope='module')
def artefactos():
    return inferencia_v2.cargar_artefactos_v2(), inferencia.cargar_artefactos()


def test_tres_bandas_y_modo_uso(artefactos):
    A2, A1 = artefactos
    r = inferencia_v2.evaluar_v2(A2, A1, PAYLOAD)
    assert set(r) == {'model_results', 'features_derivadas', 'warnings'}
    por_lender = {x['lender_id']: x for x in r['model_results']}
    assert len(por_lender) == 5
    for x in r['model_results']:
        assert x['prediction']['banda'] in (0, 1, 2)
    assert por_lender[6]['model_name'] == 'Addi_V2'
    assert por_lender[999]['model_name'] == 'General_V2'
    # BdB es solo_descarte: clase 2 se rotula revision, nunca alta
    assert por_lender[5]['prediction']['modo_uso'] == 'solo_descarte'
    assert por_lender[5]['prediction']['clases']['2'] == 'revision'
    # Meddipay se sirve con el modelo v1
    assert por_lender[39]['model_name'] == 'Meddipay_V1'
    assert 'campo_leakage_ignorado: initial_fee' in r['warnings']


def test_resto_sirve_por_tasa_primero(artefactos):
    A2, A1 = artefactos
    # Welli (23) no tiene modelo propio pero sí tasa observada con soporte:
    # debe servirse por tasa (validación post-corte), no por el General.
    r = inferencia_v2.evaluar_v2(A2, A1, {**PAYLOAD, 'lenders': [23]})
    x = r['model_results'][0]
    assert x['model_name'] == 'tasa_observada'
    assert x['prediction']['modo_uso'] == 'decision'
    assert x['prediction']['banda'] in (0, 1, 2)


def test_maxplan_sin_fee_number(artefactos):
    A2, A1 = artefactos
    p = {'lenders': [6],
         'solicitudes': [{**PAYLOAD['solicitudes'][0],
                          'solicitud': {**PAYLOAD['solicitudes'][0]['solicitud'],
                                        'fee_number': None}}]}
    r = inferencia_v2.evaluar_v2(A2, A1, p)
    pred = r['model_results'][0]['prediction']
    assert pred['mejor_plan'] in inferencia.PLANES
    assert pred['banda'] in (0, 1, 2)


def test_derivacion_nulos():
    fila, w = inferencia_v2.derivar({
        'solicitud': {'amount': 100000, 'fecha_solicitud': '2026-08-18T10:00:00'},
        'usuario': {}})
    assert fila['income_best'] is None
    assert fila['gender'] == 'N'
    assert fila['occupation'] == 'desconocida'
    assert fila['has_experian'] == 0.0
    assert fila['user_tenure_days'] == 0
    assert 'sin_bureau_de_ingresos' in w
