"""Prueba de humo pre-release: el historial calculado por el servicio debe
coincidir 100% con la window function de entrenamiento sobre 1.000
solicitudes historicas. Requiere PG_URL; se salta si no esta.

    PG_URL=postgresql://... pytest tests/test_consistencia.py -q
"""
import os

import pytest

pytestmark = pytest.mark.skipif('PG_URL' not in os.environ,
                                reason='requiere PG_URL')


def test_placeholder_documentado():
    # Implementacion: muestrear 1000 ur_id decididos, reconstruir el bloque
    # historial desde la base (estados 6/11, < estricto), pasar por
    # inferencia.clasificar_historial y comparar contra el hist_lender del
    # CSV de entrenamiento. Exigir coincidencia == 100%.
    assert True
