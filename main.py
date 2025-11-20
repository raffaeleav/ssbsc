import os

from ssbsc.swl import tune as tn
from ssbsc.swl import test as ts
from ssbsc.core import test as sts
from ssbsc.helpers import folders as fld


if __name__ == "__main__":
    fld.setup()

    temp_dir = fld.get_temp_dir()
    results_dir = fld.get_results_dir()

    model = fld.get_file_path(temp_dir, "model.safetensors")
    swl_results = fld.get_file_path(results_dir, "swl_*.json")
    ssbsc_results = fld.get_file_path(results_dir, "ssbsc_*.json")

    if os.path.isfile(model): 
        tn.tune_model()
    
    if os.path.isfile(swl_results): 
        ts.test_model()

    if os.path.isfile(ssbsc_results): 
        sts.test_model()
