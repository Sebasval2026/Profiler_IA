"""Recalcula MANIFIESTO.json: SHA-256 + filas de cada CSV de datos/."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
man = {'generado': datetime.now().isoformat(timespec='seconds'),
       'target': '11=aprobada, 6=negada, resto EXCLUIDO',
       'cortes': {'A': ['2026-04-01', '2026-05-15'],
                  'B': ['2026-03-01', '2026-04-10']},
       'datasets': {}}
for f in sorted(RAIZ.glob('datos/*.csv')) + sorted(RAIZ.glob('*.csv')):
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    n = sum(1 for _ in open(f)) - 1
    man['datasets'][f.name] = {'sha256': h, 'filas': n}
out = RAIZ / 'datos' / 'MANIFIESTO.json'
out.write_text(json.dumps(man, indent=2, ensure_ascii=False))
print('->', out)
for k, v in man['datasets'].items():
    print('  %-22s %7d filas  %s' % (k, v['filas'], v['sha256'][:16]))
