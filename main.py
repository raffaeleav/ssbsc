import os

from glob import glob
from ssbsc.helpers import folders as fld
from ssbsc.swl import test as ts, tune as tn
from ssbsc.core import test as sts, tune as sts


if __name__ == "__main__":
    fld.setup()

    swl_temp_dir = fld.get_swl_temp_dir()
    ssbsc_temp_dir = fld.get_ssbsc_temp_dir
    results_dir = fld.get_results_dir()

    swl_model = fld.get_file_path(swl_temp_dir, "model.safetensors")
    swl_results = fld.get_file_path(results_dir, "swl_*.json")

    ssbsc_model = fld.get_file_path(ssbsc_temp_dir, "model.safetensors")
    ssbsc_results = fld.get_file_path(results_dir, "ssbsc_*.json")

    if not os.path.isfile(swl_model): 
        tn.tune_model()
    
    if not glob(swl_results): 
        ts.test_model()

    if not os.path.isfile(swl_model): 
        sts.tune_model()
    
    if not glob(swl_results): 
        sts.test_model()
