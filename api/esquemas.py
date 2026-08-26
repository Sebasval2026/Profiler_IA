"""Esquemas Pydantic del contrato. Campos no listados se ignoran sin error."""
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra='ignore')


class Solicitud(_Base):
    amount: float = Field(gt=0)
    fecha_solicitud: str
    allied_id: int
    allied_industry_id: int


class Usuario(_Base):
    age: float = Field(gt=13, lt=120)
    user_created_at: str


class HistLender(_Base):
    aprobadas: int = 0
    negadas: int = 0
    fecha_ultima_decidida: Optional[str] = None


class Historial(_Base):
    # Exhaustivo por contrato: TODOS los lenders con solicitudes decididas
    por_lender: Dict[str, HistLender] = {}


class Experian(_Base):
    score: Optional[float] = None
    # los otros ~26 campos del bloque llegan y se ignoran a proposito:
    # medidos y no sobreviven la certificacion (ver README, seccion features)


class Payload(_Base):
    request_id: int
    lenders: List[int] = Field(min_length=1)
    solicitud: Solicitud
    usuario: Usuario
    historial: Optional[Historial] = None
    experian: Optional[Experian] = None
    shadow: bool = True


class Prediction(BaseModel):
    approval_band: int  # 0=baja, 1=media, 2=alta


class ModelResult(BaseModel):
    lender_id: int
    prediction: Prediction


class Respuesta(BaseModel):
    request_id: int
    model_results: List[ModelResult]
