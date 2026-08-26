# Datos de entrenamiento

Los CSV **no se versionan en git** (ver `.gitignore`). Se versionan:
la query que los genera, y el hash SHA-256 del resultado en `MANIFIESTO.json`.
Un numero reportado sin su hash de dataset no es reproducible.

| Archivo | Filas | Poblacion |
|---|---|---|
| `features_6.csv` | 51.142 | Addi, decididas desde 2025-08 |
| `features_39.csv` | 15.539 | Meddipay |
| `features_gen.csv` | 19.899 | 12 entidades del generalista |
| `tasas.csv` | 267 | tasa observada por (lender, comercio), ultimos 6 meses |

## Regenerar

```bash
psql "$PG_URL" -f extraccion/features.sql   > ../features_6.csv     # y variantes
psql "$PG_URL" -f extraccion/tasas.sql      > ../tasas.csv
python3 ../entrenamiento/manifiesto.py      # recalcula hashes
```

## Definiciones que NO se cambian sin recertificar

- **Target**: `ur_user_request_status_id` — 11 aprobada (1), 6 negada (0),
  **todo otro estado EXCLUIDO** (no es negativo: es ausencia de decision).
- **Cortes temporales**: A (train<2026-04-01, calib<2026-05-15) y
  B (train<2026-03-01, calib<2026-04-10). Validacion siempre temporal.
- **As-of estricto**: toda feature usa solo informacion anterior a
  `ur_created_at`, con `<` estricto.

## Trampas de datos (costaron dias; no las deshaga nadie)

1. El schema `migration` guarda el nulo como TEXTO literal. Todo casteo con
   guarda de regex; `NULLIF(col,'')` no lo atrapa.
2. `ex_score`: 0 y 1 son codigos, no scores. Filtrar `::int > 1`.
3. `sum() OVER (... 1 PRECEDING)` devuelve NULL (no 0) en la primera fila de
   cada particion: sin COALESCE, 42.746 primerizos caen mal clasificados.
4. `NOT (col ~ 'regex')` con col NULL descarta la fila en el WHERE.
5. Los blobs `ex_data` llegan a 1,6 MB; acotar JOINs a <250 KB.
6. La conexion a RDS exige SSL (`hostssl` en pg_hba).
