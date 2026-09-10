#!/usr/bin/env python
"""
Run a list of job configs in parallel, one OS process each.

    python runners/batch.py jobs/<name>.json [max_parallel]

Each engine runner is single-process; the Mac's cores are used by running
several jobs at once (BLAS is pinned to 2 threads per job so they don't fight).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")

RUNNER = {
    "simp": "runners/run_simp.py",
    "pytopo3d": "runners/run_pytopo3d.py",
    "topy": "runners/run_topy.py",
}


def main(spec_path, max_parallel=5):
    with open(spec_path) as f:
        jobs = json.load(f)
    tmp = os.path.join(ROOT, "runs", "_cfg")
    os.makedirs(tmp, exist_ok=True)
    logdir = os.path.join(ROOT, "runs", "_logs")
    os.makedirs(logdir, exist_ok=True)

    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2",
               NUMEXPR_NUM_THREADS="2")

    queue = list(jobs)
    running = []           # (proc, job, logfile handle, t0)
    done = []
    t_start = time.time()
    while queue or running:
        while queue and len(running) < max_parallel:
            job = queue.pop(0)
            cfg_path = os.path.join(tmp, job["id"] + ".json")
            with open(cfg_path, "w") as f:
                json.dump(job, f)
            log = open(os.path.join(logdir, job["id"] + ".log"), "w")
            runner = RUNNER[job.get("engine", "simp")]
            p = subprocess.Popen([PY, os.path.join(ROOT, runner), cfg_path],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env=env)
            running.append((p, job, log, time.time()))
            print("[start] %-28s (%s)" % (job["id"], job.get("engine", "simp")),
                  flush=True)
        time.sleep(1.0)
        for entry in list(running):
            p, job, log, t0 = entry
            if p.poll() is not None:
                running.remove(entry)
                log.close()
                status = "ok" if p.returncode == 0 else "FAILED(%d)" % p.returncode
                print("[ done] %-28s %-12s %5.0fs   (%d left)" %
                      (job["id"], status, time.time() - t0,
                       len(queue) + len(running)), flush=True)
                done.append((job["id"], p.returncode))
    bad = [j for j, rc in done if rc != 0]
    print("\n%d jobs in %.1f min; %d failed%s" %
          (len(done), (time.time() - t_start) / 60, len(bad),
           (": " + ", ".join(bad)) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5))
