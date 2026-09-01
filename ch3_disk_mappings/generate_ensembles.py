#!/usr/bin/env python3
"""
Generate the ensembles of random flat lattice disks used in Chapter 3.

Calls the C++ sampler once per disk and stores its output as a .txt file.

The sampler argument -n h fixes the minimal half-perimeter, and the
perimeter of the sampled disk fluctuates above 2h, so a target perimeter m
is obtained by keeping only the disks with |perimeter - m| <= TOLERANCE * m.

"""

import subprocess
from pathlib import Path

BINARY = "./disk_generator/disk_generator"
OUTDIR = Path("data")
N_DISKS = 500
TOLERANCE = 0.05

# target perimeter m -> minimal half-perimeter h passed to the sampler
SIZES = {50: 23, 100: 48, 200: 95, 500: 240, 1000: 450, 2000: 950}


def sample(half_perimeter):
    """
    Call the sampler once.

    Returns
    -------
    text : str
        The sampled disk, one record per line
    perimeter : int
        Number of boundary edges of the disk
    """

    output = subprocess.run(
        [BINARY, f"-n{half_perimeter}"],
        capture_output=True, text=True, check=True,
    ).stdout.split()

    # first block: n_squares, then 6 numbers per square
    n_squares = int(output[0])
    records = [output[0:1]]
    records += [output[1 + 6 * i:7 + 6 * i] for i in range(n_squares)]

    # second block: n_boundary, then 2 numbers per boundary edge
    rest = output[1 + 6 * n_squares:]
    perimeter = int(rest[0])
    records += [rest[0:1]]
    records += [rest[1 + 2 * j:3 + 2 * j] for j in range(perimeter)]

    text = "\n".join(" ".join(record) for record in records) + "\n"

    return text, perimeter


for m, h in SIZES.items():

    directory = OUTDIR / f"ensemble_m{m}"
    directory.mkdir(parents=True, exist_ok=True)

    accepted = 0
    attempts = 0

    while accepted < N_DISKS:

        text, perimeter = sample(h)
        attempts += 1

        if abs(perimeter - m) <= TOLERANCE * m:
            (directory / f"disk_{accepted:05d}.txt").write_text(text)
            accepted += 1

        if attempts > 200 * N_DISKS:
            raise RuntimeError(
                f"m = {m}: only {accepted} disks after {attempts} attempts, "
                f"the perimeter window does not match h = {h}"
            )

    print(f"m = {m}: {accepted} disks in {directory} "
          f"({accepted / attempts:.1%} accepted)", flush=True)
