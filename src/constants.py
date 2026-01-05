from pathlib import Path

# Model paths - saved models stored in repo
SAVED_MODELS_DIR = Path(__file__).parent / "saved_models"
HUBERT_DIR = SAVED_MODELS_DIR / "hubert"
RMVPE_DIR = SAVED_MODELS_DIR / "rmvpe"

# Default pretrained model paths
DEFAULT_HUBERT_PATH = str(HUBERT_DIR / "hubert_base.pt")
DEFAULT_RMVPE_PATH = str(RMVPE_DIR / "rmvpe.pt")

