"""Load CID classifier terms from a YAML registry file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - runtime dependency guard
    yaml = None


@dataclass
class CIDRegistry:
    cid_terms: List[str]
    generic_terms: List[str]
    cid_semantic_references: List[str]
    replacements: Dict[str, str]


_DEF_REGISTRY_PATH = Path(__file__).resolve().parent / "registry" / "cid_registry.yaml"


def load_cid_registry(path: Optional[str] = None) -> CIDRegistry:
    """Load CID term definitions from YAML.

    Args:
        path: Optional override path to a YAML registry.
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load registry YAML. Install with: pip install pyyaml"
        )

    registry_path = Path(path) if path else _DEF_REGISTRY_PATH

    with registry_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return CIDRegistry(
        cid_terms=list(data.get("cid_terms", [])),
        generic_terms=list(data.get("generic_terms", [])),
        cid_semantic_references=list(data.get("cid_semantic_references", [])),
        replacements=dict(data.get("replacements", {})),
    )
