"""
API del perfilador — FastAPI.

  uvicorn api.main:app --host 0.0.0.0 --port 8080

Stateless: sin conexion a base. Artefactos (modelos, politica, tasas) se
cargan una vez al arrancar desde ../modelos/.
"""
from fastapi import FastAPI, HTTPException

from . import inferencia
from .esquemas import Payload, Respuesta

app = FastAPI(title='Perfilador Creditop', version=inferencia.VERSION)
ART = None


@app.on_event('startup')
def _cargar():
    global ART
    ART = inferencia.cargar_artefactos()


@app.get('/health')
def health():
    faltan = [n for n in ('addi_central', 'addi_thin', 'meddipay_central',
                          'generalista') if ART.get(n) is None]
    return {'status': 'ok' if not faltan else 'degraded',
            'version': inferencia.VERSION, 'modelos_faltantes': faltan,
            'tasas': len(ART['tasas'])}


@app.post('/v1/profile', response_model=Respuesta)
def profile(payload: Payload):
    try:
        return inferencia.evaluar(ART, payload.model_dump())
    except KeyError as e:
        raise HTTPException(422, detail='campo faltante: %s' % e)
    except Exception as e:
        raise HTTPException(500, detail=str(e)[:200])
