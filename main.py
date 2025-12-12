import os

from glob import glob
from ssbsc.helpers import folders as fld
from ssbsc.swl import test as ts, tune as tn
from ssbsc.core.st5 import tune as stn


if __name__ == "__main__":
    fld.setup()

    swl_temp_dir = fld.get_swl_temp_dir()
    results_dir = fld.get_results_dir()

    swl_model = fld.get_file_path(swl_temp_dir, "model.safetensors")
    swl_results = fld.get_file_path(results_dir, "swl_*.json")
    ssbsc_results = fld.get_file_path(results_dir, "ssbsc_*.json")

    datasets_dir = fld.get_datasets_dir()
    ssbsc_test_pairs = fld.get_file_path(datasets_dir, "ssbsc_test_pairs.json")

    # this is needed when building test dataset to not go OOM on VRAM
    if not os.path.isfile(ssbsc_test_pairs):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # tn.tune_model()
    # ts.test_model()

    stn.tune_model()
    # sts.test_model()
