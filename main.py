import torch

from ssbsc.swl import tune as tn
from ssbsc.swl import test as ts
from ssbsc.helpers import folders as fld


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    fld.setup()

    # tn.tune_model()
    ts.test_model()
