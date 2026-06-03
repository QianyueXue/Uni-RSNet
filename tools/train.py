# by xueqianyue
import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


def main(args):
    dist.init_distributed()

    if args.seed is not None:
        dist.set_seed(args.seed)

    if args.tuning and args.resume:
        raise ValueError("Use only one of --tuning or --resume.")

    cfg_kwargs = {
        "resume": args.resume,
        "tuning": args.tuning,
        "use_amp": args.amp,
        "device": args.device,
    }
    if args.resume:
        cfg_kwargs["PResNet"] = {"pretrained": False}

    cfg = YAMLConfig(args.config, **cfg_kwargs)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Uni-RSNet training entry")
    parser.add_argument("-c", "--config", type=str, default="configs/unirsnet.yml")
    parser.add_argument("-r", "--resume", type=str, default="")
    parser.add_argument("-t", "--tuning", type=str, default="")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    main(parser.parse_args())
