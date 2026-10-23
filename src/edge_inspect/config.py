"""Strict TOML configuration with stable, config-relative paths."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import tomllib


class ValidationError(ValueError):
    """An actionable configuration or input error."""


@dataclass(frozen=True)
class Config:
    data_root: Path
    category: str
    seed: int
    artifact_root: Path
    device: str

    def as_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def load_config(path: Path) -> Config:
    path = path.expanduser().resolve()
    try:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValidationError(f"Cannot read config {path}: {exc}") from exc
    required = {"data_root", "category", "seed", "artifact_root", "device"}
    if set(values) != required:
        raise ValidationError(f"Config keys: missing={sorted(required - values.keys())}, unknown={sorted(values.keys() - required)}")
    category = values["category"]
    if not isinstance(category, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", category):
        raise ValidationError("category must be a lowercase name, e.g. bottle, without path separators")
    seed = values["seed"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValidationError("seed must be an integer between 0 and 4294967295")
    if values["device"] != "cpu":
        raise ValidationError("device must be 'cpu' for this foundation; accelerated inference is not implemented")
    paths = {}
    for key in ("data_root", "artifact_root"):
        value = values[key]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{key} must be a nonempty path string")
        candidate = Path(value).expanduser()
        candidate = (path.parent / candidate).resolve()
        if candidate.exists() and not candidate.is_dir():
            raise ValidationError(f"{key} must name a directory: {candidate}")
        paths[key] = candidate
    data, artifacts = paths["data_root"], paths["artifact_root"]
    if data == artifacts or data in artifacts.parents or artifacts in data.parents:
        raise ValidationError("data_root and artifact_root must not overlap")
    return Config(**paths, category=category, seed=seed, device=values["device"])
