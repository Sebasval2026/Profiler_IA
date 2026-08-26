"""
API del perfilador — FastAPI.

  uvicorn api.main:app --host 0.0.0.0 --port 8080

Stateless: sin conexion a base. Artefactos (modelos v1 y v2, politica, tasas)
se cargan una vez al arrancar desde ../modelos/.

Endpoints:
  POST /v1/profile  contrato minimo v1 (bandas certificadas)
  POST /v2/predict  contrato v2 hibrido (3 bandas + modo_uso por lender)
  GET  /health
"""
import os

from fastapi import FastAPI, Header, HTTPException

from . import inferencia, inferencia_v2
from .esquemas import Payload, PayloadV2, Respuesta

# ponytail: auth solo si API_KEY_PERFILADOR esta definida (prod); en dev abre
API_KEY = os.environ.get('API_KEY_PERFILADOR')

app = FastAPI(title='Perfilador Creditop', version=inferencia_v2.VERSION_V2)
ART = None
ART2 = None


@app.on_event('startup')
def _cargar():
    global ART, ART2
    ART = inferencia.cargar_artefactos()
    ART2 = inferencia_v2.cargar_artefactos_v2()


@app.get('/health')
def health():
    faltan_v1 = [n for n in ('addi_central', 'addi_thin', 'meddipay_central',
                             'generalista') if ART.get(n) is None]
    esperados_v2 = ['Addi_V2', 'BancoBogota_V2', 'Brilla_V2', 'Davivienda_V2',
                    'General_V2', 'Sistecredito_V2']
    faltan_v2 = [n for n in esperados_v2 if n not in ART2]
    return {'status': 'ok' if not (faltan_v1 or faltan_v2) else 'degraded',
            'version_v1': inferencia.VERSION,
            'version_v2': inferencia_v2.VERSION_V2,
            'modelos_faltantes': faltan_v1 + faltan_v2,
            'tasas': len(ART['tasas'])}


def _auth(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(403, detail='Acceso denegado')


@app.post('/v1/profile', response_model=Respuesta)
def profile(payload: Payload):
    try:
        return inferencia.evaluar(ART, payload.model_dump())
    except KeyError as e:
        raise HTTPException(422, detail='campo faltante: %s' % e)
    except Exception as e:
        raise HTTPException(500, detail=str(e)[:200])


@app.post('/v2/predict')
def predict_v2(payload: PayloadV2, x_api_key: str = Header(default=None)):
    _auth(x_api_key)
    try:
        return inferencia_v2.evaluar_v2(ART2, ART, payload.model_dump())
    except KeyError as e:
        raise HTTPException(422, detail='campo faltante: %s' % e)
    except Exception as e:
        raise HTTPException(500, detail=str(e)[:200])
