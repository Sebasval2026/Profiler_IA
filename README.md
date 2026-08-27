# Perfilador de aprobación de crédito — Creditop

Estima, por entidad crediticia, la probabilidad de que a un solicitante le
aprueben el crédito, en bandas: `0=baja · 1=media · 2=alta`. Corre después de
la consulta a Datacrédito y **antes** de que el usuario elija plan de cuotas.

El producto es enrutamiento: *"¿cuál entidad me aprueba?"*. Si no hay plazo
elegido, se barren los planes 6/12/18/24/36 y se toma el máximo (max-plan).

Versión actual: **v2.1-híbrido** — modelos de boosting por lender (pipeline
v2, informe 2026-08-18) servidos con la arquitectura stateless de v1.
**Todos los lenders emiten las 3 bandas**; el guardarraíl de calidad no es
suprimir bandas sino el campo `modo_uso` (`decision` / `solo_descarte` /
`ordenamiento`): donde la banda alta no es confiable, la clase 2 se rotula
`"revision"` y nunca debe usarse como pre-aprobación. El servicio expone dos
endpoints: `/v2/predict` (contrato v2, el vigente) y `/v1/profile` (contrato
mínimo v1, retrocompatible).

## Stack y lenguajes

| Capa | Tecnología |
|---|---|
| Servicio de inferencia | Python 3.10+ · FastAPI · Pydantic v2 · Uvicorn |
| Modelos v2 | CatBoost · LightGBM · XGBoost (por lender, umbrales percentiles) |
| Modelos v1 (Meddipay + fallback) | scikit-learn (logística/RF + calibración isotónica) |
| Extracción de datos | SQL (PostgreSQL / RDS, queries as-of estrictas) |
| Artefactos | `.joblib` + `bandas_config` + `columnas_entrada` + `metadata` por modelo v2; `politica.json`, `tasas.csv` (v1) |
| Tests | pytest |

> Un `.joblib` **no es portable entre versiones mayores** de su librería:
> fijar en `requirements.txt` las versiones exactas del entorno de
> entrenamiento. En macOS, LightGBM requiere `brew install libomp`.

## Estructura

```
├── api/                  servicio FastAPI, stateless (sin conexión a base)
│   ├── main.py           endpoints: POST /v2/predict · POST /v1/profile · GET /health
│   ├── esquemas.py       contratos Pydantic v1 y v2 (extra=ignore)
│   ├── inferencia.py     core v1: features, historial, max-plan, router, tasas
│   └── inferencia_v2.py  core v2: derivación de features, boosters, modo_uso
├── entrenamiento/
│   ├── entrenar.py       regenera los 4 .joblib v1 (train/calib del corte A)
│   ├── validar_fresco.py evalúa el servicio sobre decisiones post-corte (base)
│   └── manifiesto.py     hashes SHA-256 de los datasets → datos/MANIFIESTO.json
├── datos/
│   ├── README.md         definiciones, trampas de datos, cómo regenerar
│   ├── MANIFIESTO.json   versión de los datos: SHA-256 + filas de cada CSV
│   └── extraccion/       queries as-of (features, tasas) — ver "Qué NO está en el repo"
├── modelos/
│   ├── v2/<Segmento>/    por modelo: model_*.joblib, bandas_config_*.json,
│   │                     columnas_entrada_*.json, metadata_*.json
│   └── ...               v1: 4 *.joblib, politica.json, tasas.csv
└── tests/                contratos v1 y v2 + consistencia entrenamiento↔servicio
```

### Qué NO está en el repo (y dónde vive)

- `datos/*.csv` — nunca se versionan; se versiona la query que los genera y su
  hash SHA-256 en `datos/MANIFIESTO.json`.
- `entrenamiento/certificar.py` y `datos/extraccion/*.sql` — viven hoy en la
  máquina de entrenamiento; pendiente migrarlos a este repo.

## Modos de uso

### 1. Servicio de inferencia (producción)

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Los artefactos (v1 y `modelos/v2/`) ya vienen en el repo; el servicio los
carga al arrancar. Con `API_KEY_PERFILADOR` definida, `/v2/predict` exige el
header `x-api-key`; sin definir, queda abierto (dev).

- `GET /health` — estado, versiones y modelos cargados. `degraded` si falta alguno.
- `POST /v2/predict` — contrato v2: 3 bandas + `modo_uso` + umbrales por lender.
- `POST /v1/profile` — contrato mínimo v1, retrocompatible.

```bash
curl -X POST localhost:8080/v2/predict -H 'content-type: application/json' \
     -d @tests/payload_ejemplo_v2.json
```

Respuesta v2 (un resultado por lender; `banda_probabilidad` y
`proba_autorizada` son listas: una posición por solicitud del batch):

```json
{ "model_results": [
    { "lender_id": 9, "model_name": "Sistecredito_V2", "model_version": "2.0",
      "calibrated_at": "2026-08-15",
      "prediction": { "banda": 2, "banda_probabilidad": [2],
        "proba_autorizada": [0.912340],
        "clases": {"0": "baja", "1": "media", "2": "alta"},
        "umbrales": {"t_bajo": 0.424378, "t_alto": 0.863461},
        "modo_uso": "decision" } },
    { "lender_id": 5, "model_name": "BancoBogota_V2", "model_version": "2.0",
      "calibrated_at": "2026-08-14",
      "prediction": { "banda": 2, "banda_probabilidad": [2],
        "proba_autorizada": [0.747079],
        "clases": {"0": "baja", "1": "media", "2": "revision"},
        "umbrales": {"t_bajo": 0.151149, "t_alto": 0.535632},
        "modo_uso": "solo_descarte" } } ],
  "features_derivadas": { "income_best": 2800000, "amount_to_income": 1.6071,
                          "has_experian": 1, "u_age": 35.3, "...": "..." },
  "warnings": [] }
```

Reglas del contrato v2 (detalle completo en el `CONTRATO_API_PERFILADOR_V2.md`
del workspace de entrenamiento):

- **Null ≠ 0**: un bureau sin respuesta se omite (objeto completo) o va con
  campos `null`; enviar ceros falsea los flags `has_*` y los ratios.
- El cliente manda **variables crudas**; las derivadas (`income_best`,
  ratios, `user_tenure_days`, flags) las calcula el servicio y las devuelve
  en `features_derivadas`.
- `initial_fee`, `final_amount`, `amount_available`, `payment_amount` son
  **leakage post-decisión**: se ignoran y se reporta warning.
- `fee_number` es opcional (extensión híbrida): si es `null`, se barren los
  planes y se devuelve `mejor_plan`.
- `calibrated_at` con más de 45 días → warning `umbrales_desactualizados`.

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

### 4. Validación post-corte contra la base

```bash
PG_URL=postgresql://... python3 entrenamiento/validar_fresco.py 2026-08-18
```

Exporta las decisiones posteriores a la fecha, reconstruye bureaus as-of e
historial, y corre cada solicitud por el mismo camino del servicio. Ver la
sección **Validación post-corte**. Los CSV quedan en `datos/validacion/`
(gitignored: contienen ids de usuario).

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

Ejemplo completo en `tests/payload_ejemplo.json`.

## Arquitectura de decisión (v2 híbrido)

```
payload v2 ──► FastAPI (valida, ignora extras, detecta leakage)
                 │
                 ├─ deriva: u_age, user_tenure_days, income_best,
                 │          ratios (amount/debt/payment_to_income,
                 │          cc_utilization), flags has_*, 27 ex_*
                 │
                 ├─ Addi (6) ─────────► CatBoost   · decision
                 ├─ Sistecrédito (9) ─► LightGBM   · decision
                 ├─ B. de Bogotá (5) ─► CatBoost   · solo_descarte (2=revision)
                 ├─ Brilla (19) ──────► CatBoost   · ordenamiento
                 ├─ Davivienda (36) ──► CatBoost   · solo_descarte (2=revision)
                 ├─ Meddipay (39) ────► modelo v1 (RF+isotónica, max-plan,
                 │                      historial) · solo_descarte
                 └─ resto ────────────► tasa observada (lender, comercio)
                                        · decision — y General_V2 (XGBoost)
                                        de respaldo cuando no hay tasa
```

## Inferencia: cómo predice cada modelo

1. **Carga**: cada modelo v2 empaqueta 4 archivos — `model_*.joblib`
   (booster), `bandas_config_*.json` (`t_bajo`/`t_alto` percentiles +
   `calibrated_at`), `columnas_entrada_*.json` y `metadata_*.json`. Al cargar
   se extraen además las **categorías vistas en entrenamiento** (de
   `pandas_categorical` en LightGBM y `get_categories` en XGBoost): un valor
   categórico nuevo va a faltante en vez de reventar el booster.
2. **Derivación**: el cliente manda crudo; `derivar()` en
   `api/inferencia_v2.py` implementa la tabla del contrato (cascadas
   Mareigua→AgilData→reportado, ratios con guardas de nulos, NaN preservado —
   los boosters manejan faltantes nativamente, **nunca imputar a 0**).
3. **Predicción**: `predict_proba` del booster; si no hay `fee_number`, se
   barren los planes `[6,12,18,24,36]` y se toma el máximo (max-plan, v1).
4. **Banda**: `p ≤ t_bajo → 0`, `p ≥ t_alto → 2`, en medio → 1. Los umbrales
   son **por lender** (percentiles asimétricos) y se recalibran mensualmente
   sobre los últimos 90 días; el contrato no cambia al recalibrar.
5. **Guardarraíl `modo_uso`** — las 3 bandas siempre salen, pero con
   instrucción de uso: `decision` (accionable), `solo_descarte` (solo la
   banda baja es accionable; la 2 se rotula `revision`, nunca pre-aprobar),
   `ordenamiento` (ranking suave, no decisión).
6. **El "resto" se sirve por tasa observada primero** (lender, comercio, de
   v1, umbrales 0.35/0.65; con soporte `n < 50` cae a la tasa global del
   lender): en la validación post-corte le ganó al General_V2 en esa
   población — son lenders que aprueban ~95 % y la señal es la política del
   lender, no el individuo. El General_V2 queda de **respaldo** cuando no hay
   tasa (lender o comercio nuevo); sin tasa ni modelo → banda 1.

## Los modelos y sus métricas (validación out-of-time)

| Segmento | Modelo | AUC test | Acc. bandas 0y2 | Cobertura | modo_uso |
|---|---|---|---|---|---|
| Addi (6) | CatBoost | 0.816 | 0.857 | 0.55 | decision |
| Sistecrédito (9) | LightGBM | 0.887 | 0.910 | 0.64 | decision (n_test=207: vigilar) |
| Banco de Bogotá (5) | CatBoost | 0.893 | 0.766 | 0.40 | solo_descarte (prec. baja 0.99, alta 0.34) |
| GENERAL (respaldo del resto) | XGBoost | 0.912 | 0.964 | 0.48 | decision — solo cuando no hay tasa observada (ver validación) |
| Davivienda (36) | CatBoost | 0.769 | 0.784 | 0.53 | solo_descarte (n_test=96) |
| Brilla (19) | CatBoost | 0.711 | 0.887 | 0.46 | ordenamiento (tasa base 0.88: medir lift) |
| Meddipay (39) | RF v1 + isotónica | — | prec. baja 89 % (certif. v1) | — | solo_descarte |

Detalle completo en `metricas_finales_v2.csv` y metadata de cada modelo.
Entrenamiento v2: ventanas de 12m con pesos de recencia (half-life 90 días),
test = últimos 3 meses, auditoría adversarial de leakage (por eso
`ur_initial_fee` y afines están prohibidos en el payload). **PayJoy quedó sin
score** (leakage + deriva; su fallback al General también falló): cuando se
confirme su `lender_id`, agregarlo a `SIN_SCORE` en `api/inferencia_v2.py`.

Los modelos v1 (logística/RF sklearn + isotónica, features de historial por
lender, bandas certificadas por compuertas) siguen sirviendo `/v1/profile` y
el segmento Meddipay; se regeneran con `entrenamiento/entrenar.py`.

## Validación post-corte (decisiones que los modelos nunca vieron)

`entrenamiento/validar_fresco.py` evalúa el servicio completo (derivación
incluida) sobre decisiones reales posteriores a una fecha, leídas de la base:

```bash
PG_URL=postgresql://... python3 entrenamiento/validar_fresco.py 2026-08-18
```

Resultado sobre 2026-08-18..25 (n=1.400, post-corte de entrenamiento), con el
enrutamiento v2.1:

| | n | AUC | Acc. 0y2 | Prec. alta | Prec. baja | Cobertura |
|---|---|---|---|---|---|---|
| **Global** | 1.400 | **0.934** | 0.952 | 0.970 | 0.838 | 0.82 |
| Addi_V2 | 426 | 0.772 | 0.848 | 0.857 | 0.843 | 0.47 |
| tasa_observada (resto) | 909 | 0.915 | 0.981 | 0.982 | 0.955 | 0.99 |
| Meddipay_V1 | 37 | 0.829 | 0.829 | 0.966 | 0.167 ⚠️ | 0.95 |

Hallazgos que definieron el enrutamiento v2.1: la accuracy de Addi calcó la
del informe (0.848 vs 0.857 — sin overfit), y en el "resto" (tasa base ~95 %)
la **tasa observada le ganó al General_V2** sobre las mismas 814 solicitudes
(prec. alta 98,5 vs 97,1 % · cobertura 98 vs 31 % · negadas reales atrapadas
21 vs 13 de 41 · AUC 0.93 vs 0.74) — el uplift del booster sobre la tasa base
(+2 pp) no habría pasado la compuerta de certificación de v1 (≥5 pp). Por eso
el General quedó de respaldo. Sistecrédito/BdB/Davivienda/Brilla no tuvieron
volumen esa semana (n=3-14): pendiente confirmarlos con más historia. La
banda baja de Meddipay_V1 falló (1/6, n chico): vigilar en la recertificación.

## Versión de los datos

Los CSV no se versionan en git: se versiona **la query que los genera y el
hash SHA-256 del resultado** en `datos/MANIFIESTO.json`
(`entrenamiento/manifiesto.py` lo recalcula). Un número reportado sin su hash
de dataset no es reproducible.

| Dataset | Filas | Población |
|---|---|---|
| `decisiones_creditop.csv` (v2) | 259.313 | todas las decididas ago-2023→2026-08, bureaus reconstruidos (Experian 21 %, AgilData 34 %, Mareigua 26 %) |
| `features_6.csv` (v1) | 51.142 | Addi, solicitudes decididas desde 2025-08 |
| `features_39.csv` (v1) | 15.539 | Meddipay |
| `features_gen.csv` (v1) | 19.899 | 12 entidades del generalista |
| `tasas.csv` | 267 | tasa observada por (lender, comercio), últimos 6 meses |

Los modelos v2 se entrenaron sobre `decisiones_creditop.csv` (export de
`bi_analysis.mv_master_creditop_request_selected` + Mareigua reconstruido);
rango y config exactos por modelo en `modelos/v2/*/metadata_*.json`.

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

## Recertificación

**Mensual**: regenerar datasets → certificar (ventanas rodantes) → nueva
`politica.json` → `entrenar.py`. El incumbente siempre compite. En cola:
Quanto (oct 2026, 28 pp de recorrido univariado) y PayJoy (nov 2026, cuando
su historia alcance).

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
| Migrar `certificar.py` y `datos/extraccion/*.sql` a este repo | tech |
| Confirmar `lender_id` de PayJoy y agregarlo a `SIN_SCORE` (hoy caería al General, que falló para PayJoy) | datos |
| Entrenar Meddipay en el pipeline v2 (hoy sirve el modelo v1); su banda baja mostró degradación en la validación post-corte (1/6 aciertos, n chico — vigilar) | datos |
| Joins de Experian/AgilData ROTOS en `mv_master_creditop_request_selected` para agosto-2026 (0,4 % de cobertura vs 64 % en julio; los datos SÍ existen en `migration.Risk_Central_*`) — arreglar antes de la próxima recertificación | datos |
