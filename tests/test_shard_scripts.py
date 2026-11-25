import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples3" / "example_OER_path.json"


def _run(*args):
    return subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_slurm_helpers_produce_and_merge_one_shard_per_job(tmp_path):
    # Two shards are enough to exercise the one-job-per-shard path; both
    # reproduce all 16 OER paths on their own.
    for index in (2, 3):
        proc = _run(ROOT / "scripts" / "run_shard.py", INPUT, index,
                    "--output-dir", tmp_path)
        assert proc.returncode == 0, proc.stderr
    # A fixed-size array may have more tasks than shards.
    proc = _run(ROOT / "scripts" / "run_shard.py", INPUT, 99,
                "--output-dir", tmp_path)
    assert proc.returncode == 0
    assert "out of range" in proc.stdout

    pickles = sorted(tmp_path.glob("shard_*.pkl"))
    assert len(pickles) == 2
    merged_path = tmp_path / "merged.pkl"
    proc = _run(ROOT / "scripts" / "merge_shards.py",
                merged_path, *pickles)
    assert proc.returncode == 0, proc.stderr

    with open(merged_path, "rb") as fh:
        merged = pickle.load(fh)
    assert len(merged.paths) == 16
    assert [len(ind) for ind in merged.index] == [16]
    assert merged.stats["n_shards"] == 2


def test_merge_helper_accepts_legacy_pair(tmp_path):
    # Build an old-style [paths, index] pickle from a real result.
    from catmkm import load_json, run_reaction_path

    result = run_reaction_path(load_json(str(INPUT)))
    legacy = tmp_path / "legacy.pkl"
    with open(legacy, "wb") as fh:
        pickle.dump([result.paths, result.index], fh)

    merged_path = tmp_path / "merged.pkl"
    proc = _run(ROOT / "scripts" / "merge_shards.py",
                merged_path, legacy)
    assert proc.returncode == 0, proc.stderr
    with open(merged_path, "rb") as fh:
        merged = pickle.load(fh)
    assert len(merged.paths) == len(result.paths)


def test_group_array_script_runs_one_multispecies_seed_per_task(tmp_path):
    import os
    from catmkm import load_json, run_reaction_path

    expected = run_reaction_path(load_json(str(INPUT)))
    groups = tmp_path / "groups.txt"
    groups.write_text("O#+OH#\nOOH#\n")
    outdir = tmp_path / "groups"

    env = dict(os.environ)
    for index in (0, 1):
        env.update({
            "INPUT": str(INPUT),
            "GROUPS_FILE": str(groups),
            "OUTDIR": str(outdir),
            "SLURM_ARRAY_TASK_ID": str(index),
        })
        proc = subprocess.run(
            ["bash", str(ROOT / "scripts" / "slurm_path_groups.slurm")],
            cwd=ROOT, capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 0, proc.stderr

    # Out-of-range tasks are a no-op, so fixed-size arrays are safe.
    env["SLURM_ARRAY_TASK_ID"] = "99"
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "slurm_path_groups.slurm")],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0
    assert "nothing to do" in proc.stdout

    pickles = sorted(outdir.glob("group_*.pkl"))
    assert len(pickles) == 2
    merged_path = tmp_path / "merged.pkl"
    proc = _run(ROOT / "scripts" / "merge_shards.py",
                merged_path, *pickles)
    assert proc.returncode == 0, proc.stderr

    with open(merged_path, "rb") as fh:
        merged = pickle.load(fh)
    assert len(merged.paths) == len(expected.paths)
    assert merged.auto_catalytic_species == ["O#", "OH#", "OOH#"]


def test_gas_shards_can_be_listed_and_run(tmp_path):
    # --list-shards all appends the gas species after the surface ones.
    proc = _run(ROOT / "scripts" / "run_shard.py", INPUT, 4,
                "--include-gas", "--output-dir", tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "shard 4: H" in proc.stdout

    import json
    meta = json.loads((tmp_path / "shard_004.json").read_text())
    assert meta["species"] == "H"

    proc = subprocess.run(
        [sys.executable, "-m", "catmkm", "--list-shards", "all", str(INPUT)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    listed = proc.stdout.split()
    assert listed[:4] == ["O#", "O2#", "OH#", "OOH#"]
    assert {"H", "H2O", "O2"} <= set(listed)
