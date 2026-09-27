"""Runtime knobs, overridable through environment variables.

Defaults reproduce the reference run on a 16-thread / 16 GB laptop. On a large cloud
instance raise BER_JOBS (countries processed in parallel, each in its own process with
BER_THREADS // BER_JOBS threads); every country needs up to ~14 GB of RAM.

    BER_THREADS     total worker threads            (default: all logical CPUs)
    BER_JOBS        countries built in parallel     (default: 1)
    BER_MAX_DF      S1 document-frequency cut-off of the retrieval product (default: 6000)
    BER_TRAIN_FRAC  per mille of non-held-out records used for fitting     (default: 400)
"""
from __future__ import annotations

import os

N_THREADS = int(os.environ.get("BER_THREADS", os.cpu_count() or 8))
JOBS = max(1, int(os.environ.get("BER_JOBS", "1")))
MAX_DF = int(os.environ.get("BER_MAX_DF", "6000"))
TRAIN_FRAC = int(os.environ.get("BER_TRAIN_FRAC", "400"))
