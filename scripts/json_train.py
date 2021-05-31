import argparse
import logging
import json
import tempfile
import torch
import numpy as np
from pathlib import Path
import recsiam.cfghelpers as cfg
import re
import lz4


def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cmdline):

    params = json.loads(Path(cmdline.json).read_text())

    results = cfg.run_ow_exp(params, cmdline.workers)

    if cmdline.results is None:
        outfile, outfile_path = tempfile.mkstemp(prefix="json-train",
                                                 suffix=".npy.lz4")
        logging.info("storing results in {}".format(outfile_path))
    else:
        outfile_path = re.sub(r"\.npy$", "", re.sub(r"\.lz4$", "",
                                                    cmdline.results))
        outfile_path += ".npy.lz4"

    with lz4.frame.open(outfile_path,
                        mode="wb",
                        compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC) as f:
        np.save(f, (results, params))

    from recsiam.openworld import session_accuracy
    met = session_accuracy(results[0], by_step=True)
    print ("0-10:  ",met[:10].mean())
    print ("10-20: ",met[10:20].mean())
    print ("20-50: ",met[20:50].mean())
    print ("50-:    ",met[50:].mean())
    print ("10-:    ",met[10:].mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=str,
                        help="path containing the json to use")
    parser.add_argument("--results", type=str, default=None,
                        help="output file")
    parser.add_argument("-w", "--workers", type=int, default=-1,
                        help="number of joblib workers")

#       verbosity
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="triggers verbose mode")
    parser.add_argument("-q", "--quite", action='store_true',
                        help="do not output warnings")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.results is not None:
        assert Path(args.results).parent.exists()
    main(args)
