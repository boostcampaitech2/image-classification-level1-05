import os
import glob
from pathlib import Path
import random
import re

import torch

import numpy as np


class SettingsHelper:
    """
    Helper class that helps to set the enviornment.
    By default, it fixes random seed to a user defined value and chooses
    which device to use for calculation.

    Args:
        args (argparse.Namespace): Input arguments
        device (torch.device): Device to use
    """

    def __init__(self, args, device=torch.device("cuda")):
        self.args = args
        self.device = device
        self._set_seed(seed=args.seed)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def get_save_dir(self, dump=False):
        """
        A function that returns the path where models, logs, etc will be saved.
        If the directory sent along arguments already exists and dump=True,
        it dumps newly generated files over existing files.
        If dump=False, it automatically generates a new directory which has a name
        that the postfix number is increased by 1.

        Args:
            dump (bool): Whether to dump existing direcory or not
        """

        save_dir = Path(os.path.join(self.args.model_dir, self.args.name))
        if not save_dir.exists() or dump:
            return str(save_dir)
        else:
            dirs = glob.glob(f"{save_dir}*")
            matches = [re.search(rf"{save_dir.stem}(\d+)", d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            return f"{save_dir}{n}"
