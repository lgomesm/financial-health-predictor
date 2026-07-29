from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    src_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    models_dir: Path
    reports_dir: Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

settings = Settings(
    project_root=PROJECT_ROOT,
    src_dir=PROJECT_ROOT / "src",
    data_dir=PROJECT_ROOT / "data",
    raw_data_dir=PROJECT_ROOT / "data" / "raw",
    processed_data_dir=PROJECT_ROOT / "data" / "processed",
    models_dir=PROJECT_ROOT / "models",
    reports_dir=PROJECT_ROOT / "reports",
)
