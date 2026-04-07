from __future__ import annotations

import importlib

from .types import MethodName

METHOD_CLASS_PATH: dict[MethodName, tuple[str, str]] = {
    MethodName.BO: ("mountaincar.algorithms.bo", "BOAlgorithm"),
    MethodName.DR: ("mountaincar.algorithms.dr", "DRAlgorithm"),
    MethodName.BO_DR: ("mountaincar.algorithms.bo_dr", "BODRAlgorithm"),
    MethodName.DORAEMON: ("mountaincar.algorithms.doraemon", "DoraemonAlgorithm"),
    MethodName.BO_DORAEMON: ("mountaincar.algorithms.bo_doraemon", "BODoraemonAlgorithm"),
}


def normalize_method_name(value: str | MethodName) -> MethodName:
    return MethodName.from_any(value)


def list_methods() -> list[MethodName]:
    return list(METHOD_CLASS_PATH.keys())


def load_algorithm_class(method: str | MethodName):
    method_name = MethodName.from_any(method)
    module_name, class_name = METHOD_CLASS_PATH[method_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
