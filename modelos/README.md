# Artefactos de produccion (no incluidos en el zip)

Este directorio recibe los artefactos generados en la maquina de entrenamiento
(~/creditop_modelo del Mac de Sebas, versionados en su git local):

- modelo_addi_central.joblib     (logistica, alta+baja certificadas)
- modelo_addi_thin.joblib        (logistica, solo baja)
- modelo_meddipay_central.joblib (random forest, solo baja)
- modelo_generalista.joblib      (random forest, solo Su+pay baja)
- politica.json                  (bandas certificadas por segmento; artefacto GENERADO)
- tasas.csv                      (tasa observada por lender x comercio, 6 meses)

Se regeneran con:
    python3 entrenamiento/entrenar.py      # los 4 joblib
    python3 entrenamiento/certificar.py    # politica.json (vive en la maquina de entrenamiento; pendiente migrar al repo)
    psql "$PG_URL" -f datos/extraccion/tasas.sql > modelos/tasas.csv

IMPORTANTE: un .joblib de sklearn NO es portable entre versiones mayores.
Fijar en requirements.txt la version exacta del entorno de entrenamiento.
