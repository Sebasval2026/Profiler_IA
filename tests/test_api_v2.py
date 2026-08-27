"""Prueba end-to-end del contrato POR EL ENDPOINT (validación Pydantic
incluida). Requiere httpx (TestClient); se salta si no está instalado."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import inferencia_v2  # noqa: E402

try:
    from fastapi.testclient import TestClient
    from api.main import app
except (ImportError, RuntimeError):
    pytestmark = pytest.mark.skip(reason='requiere httpx para TestClient')
else:
    pytestmark = pytest.mark.skipif(not inferencia_v2.DIR_V2.exists(),
                                    reason='requiere artefactos en modelos/v2/')

PAYLOAD_V2 = json.load(open(Path(__file__).parent / 'payload_ejemplo_v2.json'))


@pytest.fixture(scope='module')
def cliente():
    with TestClient(app) as c:
        yield c


def test_health(cliente):
    h = cliente.get('/health').json()
    assert h['status'] == 'ok'
    assert h['modelos_faltantes'] == []


def test_contrato_v2_completo(cliente):
    r = cliente.post('/v2/predict', json=PAYLOAD_V2)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {'model_results', 'features_derivadas', 'warnings'}
    assert len(cuerpo['model_results']) == len(PAYLOAD_V2['lenders'])
    for x in cuerpo['model_results']:
        p = x['prediction']
        assert p['banda'] in (0, 1, 2)
        assert p['modo_uso'] in ('decision', 'solo_descarte', 'ordenamiento', 'sin_score')
        assert set(p['clases']) == {'0', '1', '2'}
        assert 't_bajo' in p['umbrales'] and 't_alto' in p['umbrales']
        assert len(p['proba_autorizada']) == len(PAYLOAD_V2['solicitudes'])
    assert cuerpo['features_derivadas']['income_best'] == 2800000


def test_thin_sin_bureaus(cliente):
    r = cliente.post('/v2/predict', json={
        'lenders': [6, 23],
        'solicitudes': [{'solicitud': {'amount': 800000,
                                       'fecha_solicitud': '2026-08-27T10:00:00',
                                       'allied_id': 26, 'allied_industry_id': 3},
                         'usuario': {'age': 24}}]})
    assert r.status_code == 200
    cuerpo = r.json()
    assert 'sin_bureau_de_ingresos' in cuerpo['warnings']
    fd = cuerpo['features_derivadas']
    assert fd['has_experian'] == 0 and fd['income_best'] is None
    for x in cuerpo['model_results']:
        assert x['prediction']['banda'] in (0, 1, 2)   # thin también decide


def test_leakage_ignorado_con_warning(cliente):
    p = json.loads(json.dumps(PAYLOAD_V2))
    p['solicitudes'][0]['solicitud']['initial_fee'] = 99999
    r = cliente.post('/v2/predict', json=p)
    assert 'campo_leakage_ignorado: initial_fee' in r.json()['warnings']


def test_amount_invalido_422(cliente):
    r = cliente.post('/v2/predict', json={
        'lenders': [6],
        'solicitudes': [{'solicitud': {'amount': -5,
                                       'fecha_solicitud': '2026-08-27T10:00:00'},
                         'usuario': {}}]})
    assert r.status_code == 422


def test_v1_sigue_funcionando(cliente):
    payload = json.load(open(Path(__file__).parent / 'payload_ejemplo.json'))
    r = cliente.post('/v1/profile', json=payload)
    assert r.status_code == 200
    assert all(m['prediction']['approval_band'] in (0, 1, 2)
               for m in r.json()['model_results'])
