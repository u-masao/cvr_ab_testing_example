import os
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import pymc3 as pm


def save_trace_and_model(trace:Any, model:Any, model_path_string: str)->None:
    # save trace and model
    save_path = Path(model_path_string)
    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "wb") as fo:
        pickle.dump((trace, model), fo)
