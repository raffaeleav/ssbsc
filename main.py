import os

from glob import glob
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

    if not os.path.isfile(model): 
        tn.tune_model()
    
    if not glob(swl_results): 
        ts.test_model()

    sbpe = True
    sts.test_model(sbpe)
