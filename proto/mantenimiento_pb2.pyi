from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PredictionRequest(_message.Message):
    __slots__ = ("features",)
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    features: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, features: _Optional[_Iterable[float]] = ...) -> None: ...

class PredictionResponse(_message.Message):
    __slots__ = ("proba_no_failure", "proba_failure")
    PROBA_NO_FAILURE_FIELD_NUMBER: _ClassVar[int]
    PROBA_FAILURE_FIELD_NUMBER: _ClassVar[int]
    proba_no_failure: float
    proba_failure: float
    def __init__(self, proba_no_failure: _Optional[float] = ..., proba_failure: _Optional[float] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("ready",)
    READY_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    def __init__(self, ready: _Optional[bool] = ...) -> None: ...
