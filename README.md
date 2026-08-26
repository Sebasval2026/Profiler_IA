# Perfilador de aprobación de crédito — Creditop

Estima, por entidad crediticia, la probabilidad de que a un solicitante le
aprueben el crédito, en bandas: `0=baja · 1=media · 2=alta`. Corre después de
la consulta a Datacrédito y **antes** de que el usuario elija plan de cuotas.

El producto es enrutamiento: *"¿cuál entidad me aprueba?"*. La semántica de
banda lo refleja: **alta = existe al menos un plan de cuotas con probabilidad
alta** (barrido sobre 6/12/18/24/36); **baja = ningún plan alcanza**;
**media = el sistema no se pronuncia**.

Versión actual del servicio y de los modelos: **v1.4** (`v1.4-contrato-min`).

## Stack y lenguajes

| Capa | Tecnología |
|---|---|
| Servicio de inferencia | Python 3.10+ · FastAPI · Pydantic v2 · Uvicorn |
| Modelos | scikit-learn (LogisticRegression, RandomForestClassifier, calibración isotónica) · pandas · joblib |
| Extracción de datos | SQL (PostgreSQL / RDS, queries as-of estrictas) |
| Artefactos | `.joblib` (modelos), `politica.json` (bandas certificadas), `tasas.csv` (fallback) |
| Tests | pytest |

> Un `.joblib` de sklearn **no es portable entre versiones mayores**: fijar en
> `requirements.txt` la versión exacta del entorno donde se entrenó
> (`python3 -c "import sklearn; print(sklearn.__version__)"`).

## Estructura

```
├── api/                  servicio FastAPI, stateless (sin conexión a base)
│   ├── main.py           endpoints: POST /v1/profile · GET /health
│   ├── esquemas.py       contrato Pydantic (extra=ignore)
│   └── inferencia.py     core: features, historial, barrido max-plan, router, sombra
├── entrenamiento/
│   ├── entrenar.py       regenera los 4 .joblib (train/calib del corte A)
│   └── manifiesto.py     hashes SHA-256 de los datasets → datos/MANIFIESTO.json
├── datos/
│   ├── README.md         definiciones, trampas de datos, cómo regenerar
│   └── extraccion/       queries as-of (features, tasas) — ver "Qué NO está en el repo"
├── modelos/              destino de artefactos: *.joblib, politica.json, tasas.csv
└── tests/                contrato + prueba de consistencia entrenamiento↔servicio
```

### Qué NO está en el repo (y dónde vive)

- `modelos/*.joblib`, `politica.json`, `tasas.csv` — se generan en la máquina
  de entrenamiento (ver `modelos/README.md`). Sin ellos el servicio arranca
  `degraded` y sirve por tasa observada.
- `datos/*.csv` — nunca se versionan; se versiona la query que los genera y su
  hash SHA-256 en `datos/MANIFIESTO.json`.
- `entrenamiento/certificar.py` y `datos/extraccion/*.sql` — viven hoy en la
  máquina de entrenamiento; pendiente migrarlos a este repo.
- `sombra.jsonl` — telemetría de runtime, va a almacenamiento persistente.

## Modos de uso

### 1. Servicio de inferencia (producción)

```bash
pip install -r requirements.txt
# copiar artefactos a modelos/ (joblib, politica.json, tasas.csv)
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

- `GET /health` — estado, versión, modelos cargados y tamaño de la tabla de
  tasas. `degraded` si falta algún `.joblib`.
- `POST /v1/profile` — evalúa todos los lenders del payload y devuelve una
  banda por lender.

```bash
curl -X POST localhost:8080/v1/profile -H 'content-type: application/json' \
     -d @tests/payload_ejemplo.json
```

Respuesta (contrato mínimo — la telemetría completa va a `sombra.jsonl`):

```json
{ "request_id": 987654,
  "model_results": [
    { "lender_id": 6, "prediction": { "approval_band": 1 } },
    { "lender_id": 9, "prediction": { "approval_band": 2 } } ] }
```

El servicio es **stateless**: no toca base de datos. Todo lo que necesita
(incluido el historial del usuario) llega en el payload; los artefactos se
cargan una sola vez al arrancar desde `modelos/`.

### 2. Entrenamiento / regeneración de artefactos

```bash
# 1. regenerar datasets desde la base (requiere PG_URL, SSL obligatorio)
psql "$PG_URL" -f datos/extraccion/features.sql > datos/features_6.csv   # y variantes
psql "$PG_URL" -f datos/extraccion/tasas.sql    > modelos/tasas.csv
python3 entrenamiento/manifiesto.py             # sella hashes en MANIFIESTO.json

# 2. certificar bandas (en la máquina de entrenamiento) → politica.json
# 3. entrenar los 4 modelos con el esquema certificado
python3 entrenamiento/entrenar.py
```

### 3. Tests

```bash
pytest tests/test_contrato.py -q                     # contrato: sin red ni base
PG_URL=postgresql://... pytest tests/test_consistencia.py -q   # pre-release, requiere base
```

`test_consistencia.py` verifica que la clasificación de historial del servicio
coincida 100 % con la window function de entrenamiento sobre 1.000 solicitudes
históricas. Se salta si no hay `PG_URL`.

### 4. Modo sombra

`shadow: true` es el default del payload: cada request queda en `sombra.jsonl`
con payload, bandas internas, probabilidad, mejor_plan y versión. **Montar ese
archivo en almacenamiento persistente.**

## Contrato de entrada

Campos no listados en `api/esquemas.py` se ignoran sin error (`extra=ignore`):
el bloque `behavior` y ~26 campos de `experian` llegan y se descartan a
propósito — se midieron y no sobreviven la certificación.

| Campo | Obligatorio | Notas |
|---|---|---|
| `request_id`, `lenders[]` | sí | `lenders` mínimo 1 |
| `solicitud.{amount, fecha_solicitud, allied_id, allied_industry_id}` | sí | `amount > 0` |
| `usuario.{age, user_created_at}` | sí | `13 < age < 120` |
| `historial.por_lender` | no | **exhaustivo por contrato** (ver abajo) |
| `experian.score` | no | 0 y 1 son códigos de Datacrédito → se tratan como null |
| `shadow` | no | default `true` |

Ejemplo completo en `tests/payload_ejemplo.json`.

## Arquitectura de decisión

```
payload ──► FastAPI (valida, ignora extras)
              │
              ├─ deriva: antig_ctop, dias_ult, hist_lender (6 niveles)
              │          score saneado (0/1 de Datacrédito = null)
              │
              ├─ Addi (6) ───► modelo propio (logística) · barrido max-plan
              │                central (con score): alta+baja · thin: solo baja
              ├─ Meddipay(39)► random forest · max-plan · solo baja (requiere score)
              ├─ Su+pay (11)─► generalista (RF) · solo baja
              └─ resto ──────► tasa observada (lender, comercio) por los
                               mismos umbrales · media si no hay datos
```

## Inferencia: cómo predice cada modelo

1. **Carga**: cada `.joblib` empaqueta `{pipe, iso, feats, umbrales, version}`
   — pipeline sklearn (imputación + encoding + modelo), calibrador isotónico,
   lista de features y umbrales.
2. **Features**: se derivan del payload en `features_para()` — industria,
   monto, edad, antigüedad en Creditop, días desde la última decisión, nivel
   de historial y score saneado.
3. **Predicción**: `pipe.predict_proba()` → probabilidad cruda →
   `iso.predict()` → probabilidad calibrada.
4. **Barrido max-plan**: el usuario aún no eligió cuotas, así que se evalúan
   los planes `[6, 12, 18, 24, 36]` y se toma el máximo (`predecir_max`). El
   generalista no usa `cuotas`.
5. **Banda**: umbrales compartidos `UA=0.65` (alta) y `UB=0.35` (baja). Solo
   se emiten las bandas **certificadas** del segmento; el resto cae a `media`.
6. **Fallback sin modelo**: la tasa observada por (lender, comercio) pasa por
   los mismos umbrales; con soporte `n < 50` cae a la tasa global del lender;
   sin datos → `media`.

**Las bandas se certifican, no se configuran.** Una banda existe solo si pasó
las compuertas — n≥30, precisión≥70 %, uplift sobre tasa base ≥5 pp — **en dos
cortes temporales**. `modelos/politica.json` es el artefacto generado que dice
qué banda emite cada segmento. Suprimir o habilitar bandas a mano está
prohibido por diseño.

## Los cuatro modelos

| Segmento | Algoritmo | Features | Bandas certificadas |
|---|---|---|---|
| addi_central | logística | industria, monto, edad, hist_lender, ex_score (+cuotas barrida) | alta + baja |
| addi_thin | logística | industria, monto, edad, antig_ctop, dias_ult, hist_lender (+cuotas) | baja |
| meddipay_central | random forest | como addi_central | baja |
| generalista | random forest | + lender_id, sin cuotas | solo Su+pay: baja |

Detalles de entrenamiento (`entrenamiento/entrenar.py`): logística
`C=0.5, max_iter=3000` con one-hot (`min_frequency=30`) y estandarización;
random forest `n_estimators=300, min_samples_leaf=15, random_state=42` con
ordinal encoding. Imputación por mediana **con indicador de faltante** en
ambos. Calibración isotónica sobre la ventana de calibración del corte A.

Métricas de certificación (test temporal, ambos cortes): Addi central
prec_alta 78,8/78,1 % · prec_baja 89,7/89,3 % · banda media ≈ 36 %. El detalle
por segmento y corte vive en `modelos/politica.json`.

**Por qué logística y no boosting:** se barrieron 4 algoritmos por segmento;
en Addi la logística tiene *menor* error que los árboles (16,0 vs 20-22 %) y
en Meddipay gana el random forest por 6 pp. El algoritmo es un hiperparámetro
del segmento y el certificador lo re-decide cada mes.

## Versión de los datos

Los CSV no se versionan en git: se versiona **la query que los genera y el
hash SHA-256 del resultado** en `datos/MANIFIESTO.json`
(`entrenamiento/manifiesto.py` lo recalcula). Un número reportado sin su hash
de dataset no es reproducible.

| Dataset | Filas | Población |
|---|---|---|
| `features_6.csv` | 51.142 | Addi, solicitudes decididas desde 2025-08 |
| `features_39.csv` | 15.539 | Meddipay |
| `features_gen.csv` | 19.899 | 12 entidades del generalista |
| `tasas.csv` | 267 | tasa observada por (lender, comercio), últimos 6 meses |

Definiciones que **no se cambian sin recertificar** (detalle en
`datos/README.md`):

- **Target**: `ur_user_request_status_id` — 11 aprobada (1), 6 negada (0),
  todo otro estado **excluido** (no es negativo: es ausencia de decisión).
- **Cortes temporales** (validación siempre temporal):
  corte **A** — train < 2026-04-01, calibración < 2026-05-15;
  corte **B** — train < 2026-03-01, calibración < 2026-04-10.
  Los modelos de producción v1.4 se entrenan con el esquema del corte A.
- **As-of estricto**: toda feature usa solo información anterior a
  `ur_created_at`, con `<` estricto.

`datos/README.md` documenta además las trampas de datos conocidas (nulos como
texto literal, códigos 0/1 en `ex_score`, NULLs de window functions, etc.).

## El historial: la parte delicada del contrato

El servicio no tiene base. El cliente manda conteos primitivos por lender —
**exhaustivo: todos los lenders donde el usuario tiene solicitudes decididas**:

```json
"historial": { "por_lender": {
  "6":  { "aprobadas": 1, "negadas": 1, "fecha_ultima_decidida": "2026-07-04T16:20:00-05:00" },
  "23": { "aprobadas": 2, "negadas": 0, "fecha_ultima_decidida": "2026-05-11T09:00:00-05:00" } } }
```

Generado con exactamente esta query (los dos filtros no son negociables):

```sql
SELECT ur_lender_id,
       count(*) FILTER (WHERE ur_user_request_status_id = 11) AS aprobadas,
       count(*) FILTER (WHERE ur_user_request_status_id = 6)  AS negadas,
       max(ur_created_at)                                     AS fecha_ultima_decidida
FROM user_requests
WHERE u_id = :user_id
  AND ur_user_request_status_id IN (6, 11)   -- solo decididas
  AND ur_created_at < :fecha_solicitud       -- < estricto: no se cuenta a sí misma
GROUP BY 1
```

La clasificación a 6 niveles (`primera_vez` … `negado_2mas_aqui`) vive en
`api/inferencia.py` y **debe** coincidir con la window function de
entrenamiento: `tests/test_consistencia.py` lo verifica sobre 1.000 casos
históricos antes de cada release.

## Sombra y recertificación

- Salida de sombra (4-6 semanas): precisión por banda dentro de ±3 pp del
  holdout y calibración ≤5 pp, dos semanas consecutivas.
- **Recertificación mensual**: regenerar datasets → certificar (ventanas
  rodantes) → nueva `politica.json` → `entrenar.py`. El incumbente siempre
  compite. En cola: Quanto (oct 2026, 28 pp de recorrido univariado) y
  PayJoy (nov 2026, cuando su historia alcance).

## Decisiones cerradas con medición (no reabrir sin dato nuevo)

`cuotas` no existe en inferencia → barrido max-plan · ingreso/ocupación
declarados: no certifican, y **la presencia del dato está contaminada** (los
negados declaran más; toda feature de formulario se prueba sin indicador de
faltante) · `hora`/`día`: composición de comercios · ciudad: absorbida por el
aliado · el generalista no personaliza dentro de un lender (interacciones sin
datos) · `ur_final_amount` y afines: fuga por construcción.

## Pendientes con dueño

| Pendiente | Dueño |
|---|---|
| Rotar contraseña de postgres + rol read-only (expuesta en chat y en `ps`) | infra |
| Costo relativo falso-"aplicá" vs "no sé" → umbrales dejan de ser 0,65/0,35 a dedo | negocio |
| ¿Qué pasó con el volumen de Sistecrédito post-mayo? ¿Qué cambió con DENTIX? | negocio |
| Auditoría as-of de los flags `*_verified` (desbloquea Meddipay mejorado, −1,3 pp) | datos |
| `sombra.jsonl` a almacenamiento persistente | tech |
| Migrar `certificar.py` y `datos/extraccion/*.sql` a este repo | tech |
