# catmkm

`catmkm` is a refactor of the legacy `catrap` scripts into a reusable Python
library.  It provides:

* steady-state microkinetic rate calculations (`write_F_and_J_function`,
  `single_point_calculation`),
* degree-of-rate-control analysis,
* gas-phase thermodynamic corrections (Shomate and ideal-gas),
* reaction-path search with automatic detection of auto-catalytic species,
* scaling-relation and activity/mechanism-map plotting,
* continuous stirred-tank reactor (CSTR) simulations with evolving gas
  partial pressures.

The electrochemical PNP solver (`mkm_elechem`), the pure-phase equilibrium
solver (`mkm_sol_equ`) and external-format converters (for example
`read_from_catkinas_input`) were intentionally not ported.

## Installation

```bash
pip install -e .            # core
pip install -e .[plot]      # with matplotlib
pip install -e .[dev]       # with pytest
```

## Python API

```python
import numpy as np

from catmkm import (
    build_model, build_rate_functions, get_rate_constants,
    single_point_calculation, random_guess, correct_barriers_with_gas,
)

obj = {
    "temperature": [500],
    "reactions": ["2*# + CO <-> CO##", "2*# + O2 <-> 2*O#",
                  "CO## + O# <-> CO2 + 3*#"],
    "barriers": [[0, -0.22312], [0.82186, -1.28814], [0.72, -1.575498]],
    "gas_info": [
        {"name": "CO",  "pressure": [1e-10]},
        {"name": "O2",  "pressure": [1e-10]},
        {"name": "CO2", "pressure": [1e10]},
    ],
}

mkm = build_model(obj)
build_rate_functions(mkm)                 # in-memory code generation

T = 500
mkm.gas = np.array([g.pressure[0] for g in mkm.gas_list])
barriers = np.array(obj["barriers"], dtype=float)
correct_barriers_with_gas(mkm, barriers, T, None, obj["gas_info"])
mkm.rate_constants = get_rate_constants(mkm, barriers, T, 0)
mkm.rate_constants_mpf = get_rate_constants(mkm, barriers, T, mkm.data_ndigit)

y0 = random_guess(mkm.normalize_cvr, mkm.constrain_list)
coverage, rates = single_point_calculation(
    {"ode": {"ode_eps": 1e-6, "ode_nmax": 30},
     "newton": {"newton_eps": "1.0e-100", "newton_nmax": 30}},
    y0, mkm)
```

`build_rate_functions` compiles the generated `calc_F`/`calc_J` directly in
memory.  If you need an actual source file, use `write_F_and_J_function(mkm,
"F_and_J.py")` instead.

The high-level helper `run_input(dict)` / `run_file(path)` dispatches to the
calculation described by the input and returns Python objects.

## Command line

```bash
catmkm input.json -o output.json
python -m catmkm input.json -o output.json
```

A reaction-path result is pickled; steady-state and map results are written as
JSON.  The calculation mode is auto-detected but can be overridden with
`--mode {steady-state,reaction-path,activity-map,cstr}`.

Reaction-path sharding can also be controlled from the command line:

```bash
# List the available shard species (add "all" to include gas species).
catmkm paths.json --list-shards
catmkm paths.json --list-shards all

# Run one shard (one Slurm job).
catmkm paths.json --shard "CO#" -o co.pkl

# Run every surface intermediate shard, in parallel, and merge the result.
catmkm paths.json --shard each --jobs 8 -o merged.pkl

# Legacy-style enumeration: surface intermediates plus every gas species.
catmkm paths.json --shard all --jobs 8 -o merged_all.pkl

# Multi-species seed group ("+" joins species, ";" separates groups).
catmkm paths.json --shard "CH3#+CO#+H#" -o group.pkl
```

## Input format

Common fields:

| field | meaning |
| --- | --- |
| `temperature` | number or list of temperatures |
| `reactions` | elementary reaction strings |
| `barriers` | `[[Ea, H], ...]` or `[{"name": ..., "data": [...]}, ...]` |
| `gas_info` | list of `{"name": ..., "pressure": [...]}` |
| `set_multi_site` | e.g. `{"#1": 0.95, "#2": 0.05}` |
| `ode_setting`, `newton_setting` | numerical settings |
| `drc_delta_e` | DRC perturbation; non-positive disables DRC |
| `thermo_correction`, `gas_correct_type` | gas thermodynamic corrections |
| `random_seed` | optional seed for randomised path searches |
| `cstr` | CSTR settings (see below) |

### Reaction-path input (clean form)

Barriers, temperature and gas pressures are not required for a path search:

```json
{
  "reactions": ["O2 + # <-> O2#", "O2# + # <-> 2*O#", "..."],
  "total_reactions": ["O2 + 4*H <-> 2*H2O"]
}
```

Auto-catalytic species (species that are both consumed and produced) are now
detected automatically.  An explicit `auto_catalytic_species` list overrides
the detection and `[]` disables it.  Gas species are allowed in that list; a
gas product such as `CH3OH` or `CH4` lets the reverse of its formation steps
act as a source of surface species, which is how the legacy per-species jobs
reached part of the path set.

### Sharded reaction-path search

For large networks the same search can be run once per seed group and the
results merged.  With one species per group this is the old “one Slurm job per
intermediate, combine the pickles afterwards” workflow, now available
directly:

```json
{
  "reactions": ["..."],
  "total_reactions": ["..."],
  "find_reaction_path": {
    "shard_species": "each",
    "merge": true,
    "n_jobs": 8,
    "find_reverse": true,
    "check_rank": true,
    "save_depth": 10
  }
}
```

`shard_species` accepts `"each"` (all non-bare surface intermediates), `"all"`
(the same plus every gas species), `"auto"` (only detected auto-catalytic
species), a single species name, or an explicit list.  Gas species are valid
seeds: the legacy pipeline also ran one job per gas species, and those runs do
add paths on ex1 (`CH3OH`, for instance, recovers a large part of the gap
described below).  `merge` removes duplicate paths (same total reaction,
reaction support and direction) and sorts the result deterministically.
`n_jobs` runs the shards in parallel in one process; for a cluster, run one
shard per job with `--shard` and merge separately with
`merge_reaction_paths`.

A list of lists gives explicit **multi-species seed groups**, one search per
group:

```json
"shard_species": [["CH3#", "CO#", "H#"], ["O#", "OH#"]]
```

This matters for compatibility with the legacy results.  A one-species-per-job
union over the surface intermediates alone finds 1864 of the 2466 paths in the
legacy `catrap_path.pkl`.  Adding the seven gas species (`shard_species="all"`)
raises that to 2050; only `CH3OH` (+186 paths), `CH4` (+13) and `H2O` (+8)
contribute at `save_depth=10`, while `CH2CO`, `CO2`, `CO` and `H2` add nothing.
A single search with both legacy seeds, `auto_catalytic_species=["CH3OH",
"CH2CO"]`, finds 2294 paths and together with the per-species union covers
2337 of the 2466 (the two codes agree exactly for that seed set: same RNG,
1235/1235 identical at a 2 M-node cap).  The remaining 131 need still larger
initial-state sets; use multi-species groups or seed the detected
auto-catalytic list in a single search when those pathways are wanted.  For
identical seed sets the new code and the legacy code return exactly the same
paths (empty seeds on the full ex1 network: 253/253 identical; gas-seeded
`CH3OH` with the same RNG: 613/613 identical; restricted 9-reaction test:
identical).

From Python:

```python
from catmkm import (
    build_model, find_reaction_paths_sharded, list_shard_species,
    merge_reaction_paths,
)

mkm = build_model({"reactions": reactions}, for_path=True)
mkm_gas = build_model({"reactions": total_reactions}, for_path=True)

print(list_shard_species(mkm))
print(list_shard_species(mkm, include_gas=True))
result = find_reaction_paths_sharded(
    mkm, mkm_gas, shards="each", n_jobs=8, merge=True,
    find_reverse=True, check_rank=True, save_depth=10,
)

# Legacy-style enumeration: surface intermediates and gas species.
result = find_reaction_paths_sharded(
    mkm, mkm_gas, shards="all", n_jobs=8, merge=True,
    find_reverse=True, check_rank=True, save_depth=10,
)

# Explicit multi-species seed groups:
result = find_reaction_paths_sharded(
    mkm, mkm_gas, shards=[["CH3#", "CO#", "H#"], ["O#", "OH#"]],
    n_jobs=2, find_reverse=True, check_rank=True, save_depth=10,
)
```

`shards=...` with `merge=False` returns the list of per-shard results;
`merge_reaction_paths([...])` combines them later (for example after loading
the pickles produced by separate Slurm jobs).  Old-style pickles containing
``[paths, index]`` are accepted directly by `merge_reaction_paths`:

```python
import pickle
from catmkm import merge_reaction_paths

results = [pickle.load(open(name, "rb")) for name in pickle_files]
merged = merge_reaction_paths(results)
```

### CSTR input

```json
{
  "temperature": [500],
  "reactions": ["..."],
  "barriers": [[...]],
  "gas_info": [{"name": "CO", "pressure": [1.0]}, "..."],
  "cstr": {
    "feed": {"CO": 1.0, "O2": 0.33, "CO2": 0.0},
    "residence_time": 1.0,
    "gas_factor": 1.0,
    "max_time": 1000.0
  }
}
```

The model is built with gas partial pressures as state variables.  The gas
balances are

```
dp_i/dt = (p_i^feed - p_i)/tau + gas_factor * sum_j nu_ij R_j
```

and the coupled system is integrated to steady state, then polished by Newton
iteration.

## Plotting

`catmkm.plotting` contains the functions from `draw_utils.py`
(`fit_scaling_relationship`, `draw_scaling_1d`, `draw_scaling_2d`,
`find_best_path`, `draw_1d_map`, `draw_2d_map`) plus
`draw_scaling_and_map(data, settings)`, the former driver script.
Matplotlib is imported lazily.

### Running the shards on Slurm

`scripts/run_shard.py` runs exactly one shard and writes one pickle, so it maps
directly onto a Slurm job array.  Each task is independent and only needs one
core.

```bash
cd /path/to/catmkm
export PYTHONPATH=$PWD:$PYTHONPATH

# How many array tasks?  (ex1 has 19 surface intermediates and 7 gas species.)
catmkm catrap/draw_rpd_examples/ex1/input_catrap.json --list-shards all | nl -v 0

# Legacy-style run: one job per species, surface and gas (26 shards on ex1).
INPUT=$PWD/catrap/draw_rpd_examples/ex1/input_catrap.json \
OUTDIR=$PWD/path_shards INCLUDE_GAS=1 \
sbatch --array=0-25 scripts/slurm_path_array.slurm

# Surface intermediates only (19 shards on ex1).
INPUT=$PWD/catrap/draw_rpd_examples/ex1/input_catrap.json \
OUTDIR=$PWD/path_shards \
sbatch --array=0-18 scripts/slurm_path_array.slurm

# Reproduce the legacy pickle: one job per species (gas included) with the
# legacy search default save_depth=2.
INPUT=$PWD/catrap/draw_rpd_examples/ex1/input_catrap.json \
OUTDIR=$PWD/path_shards INCLUDE_GAS=1 SAVE_DEPTH=2 \
sbatch --array=0-25 scripts/slurm_path_array.slurm

# Merge once every array task succeeded.
OUTDIR=$PWD/path_shards \
sbatch --dependency=afterok:<array_job_id> scripts/slurm_path_merge.slurm
```

`merge_shards.py --legacy` writes a plain `[paths, index]` pickle instead of a
`ReactionPathResult`, matching the old `catrap_path.pkl` layout for existing
plotting scripts.

Out-of-range array tasks exit successfully, so a fixed `--array=0-63` is also
safe.  The scripts use `python3` from `PATH`; load your module/conda
environment in the job script if the cluster does not provide it.  Request
enough memory per task (`--mem=8G` in the example is comfortable now that the
visited-set keys are packed integers) and a generous wall time for the largest
shards.

The same shards can be run without Slurm, either serially or with processes:

```bash
python scripts/run_shard.py paths.json 0 --output-dir path_shards
python scripts/merge_shards.py path_shards/merged.pkl path_shards/shard_*.pkl

# Include the gas species in the index space (``--list-shards all``):
python scripts/run_shard.py paths.json 19 --include-gas --output-dir path_shards

# Or let catmkm manage the worker pool (memory = n_jobs x one shard):
catmkm paths.json --shard all --jobs 6 -o merged.pkl
```

`run_shard.py` is index-based (one species per task).  For multi-species seed
groups, run one job per group with the CLI and merge the pickles:

```bash
GROUP="CH3#+CO#+H#"
catmkm "$INPUT" --shard "$GROUP" --quiet -o "$OUTDIR/group_0.pkl"
python scripts/merge_shards.py "$OUTDIR/merged.pkl" "$OUTDIR"/group_*.pkl
```

For a cluster, put the groups in a file (one group per line, `+` joining
species) and submit one array task per line:

```bash
printf 'CH3#+CO#+H#\nO#+OH#\n' > groups.txt
INPUT=$PWD/input.json GROUPS_FILE=$PWD/groups.txt OUTDIR=$PWD/path_groups \
sbatch --array=0-1 scripts/slurm_path_groups.slurm
OUTDIR=$PWD/path_groups \
sbatch --dependency=afterok:<array_job_id> scripts/slurm_path_merge.slurm
```

`slurm_path_merge.slurm` merges both `shard_*.pkl` and `group_*.pkl`.

## Performance notes

The exhaustive search runs at roughly 10⁴ search nodes per second in pure
Python.  A path search with 10⁵ paths typically takes tens of seconds and
about 1.5 KB of memory per stored path.  Three practical limits matter more
than the language:

* **Branching**: sharding by intermediate (`shard_species="each"`) is a strong
  pruning strategy as well as a way to parallelise.  Running all intermediates
  as one combined seed set can be much slower.
* **Parallelism**: `n_jobs` uses processes, so the wall time scales nearly
  linearly on a multi-core node.  On Slurm, submit one `--shard SPECIES` job
  per intermediate and merge the pickles with `merge_reaction_paths`.
* **Output size**: for very large searches use `store_paths=False` together
  with `on_path=...` to stream paths to disk instead of keeping them in RAM.

Measured on the 33-reaction ex1 network with `find_reverse=True`,
`check_rank=True`, `save_depth=10` and no node cap:

| shard | nodes | unique paths | time | peak RSS |
|---|---:|---:|---:|---:|
| `CO#` | 51 k | 253 | 3.7 s | 92 MB |
| `CH3CO#` | 6.64 M | 1 630 | 8 min 20 s | 302 MB |
| `CH3OH` (gas) | 2.64 M | 1 670 | 4 min 5 s | ~300 MB |

The earlier `frozenset` visited keys reached 2.6 GB on `CH3CO#` and were
OOM-killed in 7 GB; packing each key into one integer (two bits per reaction)
brought that shard down to ~300 MB.  For an ex1 array that includes the gas
species (26 tasks), plan one core and ~1 GB per task, with 24 h wall time
covering the slowest shard.

`save_depth` controls how many recent reaction suffixes are treated as
visited.  Lower values explore far more nodes but, in our measurements, did
not unlock the legacy-only paths: `CO#` used 1.28 M nodes at 2 vs 51 k at 10
and returned the same 253 paths; `H2O` used 3.58 M vs 159 k nodes and returned
the same ~490 paths.  The legacy code's default is 2; the old per-species job
files may have used that value.

Gas seeds matter.  On ex1, `CH3OH` alone recovers 186 of the 602 paths that
the 18 surface-only shards miss; `CH4` adds 13 and `H2O` adds 8, while `CO2`,
`CO`, `H2` and `CH2CO` add none at `save_depth=10`.  The merged union of the
surface and gas shards is 2050 of the 2466 legacy paths, with no extra paths.
Multi-species seed groups reach further: for example `{CH3#, OH#}` recovers 92
of the 602 legacy-only paths within a 2 M-node cap.

The core walker is still Python/NumPy.  A C++/pybind11 kernel with bitsets and
an incremental rank basis is the next step if searches regularly exceed a few
million paths; the exponential growth of the number of paths itself is not
removed by changing language.

## Examples

Small self-contained inputs live in `examples/`:

```bash
catmkm examples/reaction_path.json -o paths.pkl
catmkm examples/cstr.json -o cstr.json
```

The legacy test cases remain in `examples3/` and are used by the test suite.

## Tests

```bash
pytest
```

The tests compare the refactored steady-state, multi-site and activity-map
results with the values produced by the original scripts.

## Differences from the old scripts

* Errors are raised as `InputError`/`ValueError` instead of calling `exit()`.
* The ODE/Jacobian can be compiled in memory; no `F_and_J.py` round-trip.
* Auto-catalytic species are detected automatically for path searches.
* Gas partial pressures can be evolved in a CSTR.
* Electrochemistry, solution equilibrium and external-format readers are
  omitted; inputs requesting them raise a clear error.
