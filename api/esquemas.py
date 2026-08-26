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


# ------------------------------------------------------------- contrato v2

class SolicitudV2(_Base):
    # extra='allow' para detectar campos de leakage y advertir (no usarlos)
    model_config = ConfigDict(extra='allow')
    amount: float = Field(gt=0)
    fee_number: Optional[int] = Field(default=None, ge=1)  # null → barrido max-plan
    fecha_solicitud: str
    allied_id: Optional[int] = None       # extensión híbrida: fallback por tasa
    allied_industry_id: Optional[int] = None
    allied_type_id: Optional[int] = None
    credit_line_id: Optional[int] = None


class UsuarioV2(_Base):
    date_of_birth: Optional[str] = None
    age: Optional[float] = None
    gender: Optional[str] = None
    user_created_at: Optional[str] = None


class Mareigua(_Base):
    average_income: Optional[float] = None
    average_income_reported: Optional[float] = None
    continuity: Optional[float] = None
    continuity_reported: Optional[float] = None
    occupation: Optional[str] = None


class Agildata(_Base):
    average_income: Optional[float] = None
    continuity: Optional[float] = None
    age: Optional[float] = None
    genre: Optional[str] = None


class SolicitudItemV2(_Base):
    solicitud: SolicitudV2
    usuario: UsuarioV2
    mareigua: Optional[Mareigua] = None
    agildata: Optional[Agildata] = None
    # los 27 campos numéricos ex_*; validados como numéricos-o-null
    experian: Optional[Dict[str, Optional[float]]] = None
    historial: Optional[Historial] = None  # extensión híbrida: lo usa Meddipay v1


class PayloadV2(_Base):
    lenders: List[int] = Field(min_length=1)
    solicitudes: List[SolicitudItemV2] = Field(min_length=1)


class Prediction(BaseModel):
    approval_band: int  # 0=baja, 1=media, 2=alta


class ModelResult(BaseModel):
    lender_id: int
    prediction: Prediction


class Respuesta(BaseModel):
    request_id: int
    model_results: List[ModelResult]
