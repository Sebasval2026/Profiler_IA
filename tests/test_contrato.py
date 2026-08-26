"""Tests del contrato. Corren sin red y sin base: pytest tests/"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api import inferencia  # noqa: E402

PAYLOAD = {
    'request_id': 987654, 'lenders': [6, 9, 5],
    'solicitud': {'amount': 4500000, 'fecha_solicitud': '2026-08-18T10:30:00-05:00',
                  'allied_id': 26, 'allied_industry_id': 3},
    'usuario': {'age': 32, 'user_created_at': '2025-11-02T09:15:00-05:00'},
    'historial': {'por_lender': {
        '6': {'aprobadas': 1, 'negadas': 1,
              'fecha_ultima_decidida': '2026-07-04T16:20:00-05:00'},
        '9': {'aprobadas': 0, 'negadas': 0}}},
    'experian': {'score': 742},
}


import pytest

requiere_artefactos = pytest.mark.skipif(
    not (inferencia.DIR_MODELOS / 'politica.json').exists(),
    reason='requiere artefactos en modelos/ (ver modelos/README.md)')


@requiere_artefactos
def test_contrato_minimo():
    A = inferencia.cargar_artefactos()
    r = inferencia.evaluar(A, PAYLOAD)
    assert set(r) == {'request_id', 'model_results'}
    assert len(r['model_results']) == 3
    for mr in r['model_results']:
        assert mr['prediction']['approval_band'] in (0, 1, 2)


def test_historial_niveles():
    from datetime import datetime
    fs = datetime.fromisoformat('2026-08-18T10:30:00')
    hl, _ = inferencia.clasificar_historial({'por_lender': {}}, 6, fs)
    assert hl == 'primera_vez'
    hl, _ = inferencia.clasificar_historial(
        {'por_lender': {'23': {'aprobadas': 2, 'negadas': 0}}}, 6, fs)
    assert hl == 'nunca_aqui'   # historia en OTRO lender: no es primera vez
    hl, _ = inferencia.clasificar_historial(
        {'por_lender': {'6': {'aprobadas': 0, 'negadas': 2}}}, 6, fs)
    assert hl == 'negado_2mas_aqui'


def test_score_saneado():
    assert inferencia.sanear_score(742) == 742
    assert inferencia.sanear_score(0) is None    # codigo, no score
    assert inferencia.sanear_score(1) is None
    assert inferencia.sanear_score('\\N') is None
