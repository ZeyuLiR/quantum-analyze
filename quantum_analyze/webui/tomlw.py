"""Minimal TOML writer for the config files the UI saves (stdlib has only a reader)."""
from __future__ import annotations

import json
import math
import re

_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE.match(k) else json.dumps(k, ensure_ascii=False)


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("TOML 不支持 NaN/Infinity")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(x)}" for k, x in v.items()) + " }"
    if v is None:
        raise ValueError("TOML 不支持空值")
    raise TypeError(f"无法写入 TOML 的类型: {type(v).__name__}")


def dumps(data: dict, header: str = "") -> str:
    lines: list[str] = []
    if header:
        lines += [f"# {h}" for h in header.splitlines()] + [""]

    def emit(table: dict, prefix: list[str]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict) and v is not None}
        subs = {k: v for k, v in table.items() if isinstance(v, dict)}
        if prefix and (scalars or not subs):
            lines.append(f"[{'.'.join(_key(p) for p in prefix)}]")
        for k, v in scalars.items():
            lines.append(f"{_key(k)} = {_value(v)}")
        if prefix and (scalars or not subs):
            lines.append("")
        for k, v in subs.items():
            emit(v, prefix + [k])

    emit(data, [])
    return "\n".join(lines).rstrip() + "\n"
