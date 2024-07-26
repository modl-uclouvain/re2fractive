import os
from pathlib import Path

CAMPAIGN_ID = os.environ.get("RE2FRACTIVE_CAMPAIGN_ID", "0001")
CAMPAIGNS_DIR = Path(__file__).parent.parent.parent / "campaigns"
DATASETS_DIR = CAMPAIGNS_DIR / CAMPAIGN_ID / "datasets"
FEATURES_DIR = CAMPAIGNS_DIR / CAMPAIGN_ID / "features"
SCRATCH_DIR = CAMPAIGNS_DIR / CAMPAIGN_ID / "scratch"
MODELS_DIR = CAMPAIGNS_DIR / CAMPAIGN_ID / "models"
EPOCHS_DIR = CAMPAIGNS_DIR / CAMPAIGN_ID / "epochs"
RESULTS_DIR = CAMPAIGNS_DIR / CAMPAIGN_ID / "results"

__all__ = (
    "CAMPAIGNS_DIR",
    "DATASETS_DIR",
    "FEATURES_DIR",
    "CAMPAIGN_ID",
    "EPOCHS_DIR",
    "SCRATCH_DIR",
    "MODELS_DIR",
)
