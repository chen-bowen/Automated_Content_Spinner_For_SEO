import pathlib

import regression_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(content_spinner.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "models" / "model_file"