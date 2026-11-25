"""Command-line interface for catmkm."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys

from .io import build_model, detect_mode, load_json, run_input, save_result
from .reaction_path import (
    ReactionPathResult,
    list_shard_species,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="catmkm",
        description="Microkinetic calculations from a JSON input file.",
    )
    parser.add_argument("input", help="Path to the input JSON file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Write the result to this file (JSON, or pickle "
                             "for reaction paths).")
    parser.add_argument("-m", "--mode", default=None,
                        choices=["steady-state", "steady", "reaction-path",
                                 "path", "activity-map", "map", "cstr"],
                        help="Override the automatically detected mode.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the summary printed to stdout.")
    parser.add_argument("--shard", default=None, metavar="SPECIES",
                        help="For reaction-path searches, run selected "
                             "shards: a species name, a comma-separated list "
                             "(one shard per species), 'A+B' for one "
                             "multi-species seed group, ';' to separate "
                             "groups, 'each' for all intermediates, 'all' "
                             "for intermediates plus gas species, or 'auto' "
                             "for detected auto-catalytic species.")
    parser.add_argument("--list-shards", nargs="?", const="surface",
                        choices=("surface", "all"), default=None,
                        metavar="{surface,all}",
                        help="Print the available reaction-path shard species "
                             "and exit (default: surface intermediates; 'all' "
                             "also lists gas species).")
    parser.add_argument("--jobs", type=int, default=1, metavar="N",
                        help="Number of processes used by --shard each/all/"
                             "auto.")
    parser.add_argument("--no-merge", action="store_true",
                        help="Do not merge shard results (mostly useful with "
                             "the Python API).")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        obj = load_json(args.input)
        mode = args.mode or detect_mode(obj)

        if args.list_shards:
            if mode not in ("reaction-path", "path"):
                raise ValueError("--list-shards is only valid for reaction-"
                                 "path inputs.")
            mkm = build_model(obj, for_path=True)
            include_gas = args.list_shards == "all"
            for species in list_shard_species(mkm, include_gas=include_gas):
                print(species)
            return 0

        if args.shard is not None:
            if mode not in ("reaction-path", "path"):
                raise ValueError("--shard is only valid for reaction-path "
                                 "inputs.")
            fp = obj.setdefault("find_reaction_path", {})
            if args.shard in ("each", "all", "auto"):
                fp["shard_species"] = args.shard
            elif "+" in args.shard or ";" in args.shard:
                # "A+B" is one multi-species seed group; ";" separates groups.
                fp["shard_species"] = [
                    [s.strip() for s in part.split("+") if s.strip()]
                    for part in args.shard.split(";") if part.strip()
                ]
            else:
                # Backwards compatible: a comma-separated list is one shard
                # per species.
                fp["shard_species"] = [
                    [s.strip()] for s in args.shard.split(",") if s.strip()
                ]
            fp["merge"] = not args.no_merge
            fp["n_jobs"] = args.jobs

        if args.quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_input(obj, mode=mode)
        else:
            result = run_input(obj, mode=mode)

        if args.output:
            save_result(result, args.output)
            if not args.quiet:
                print(f"Result written to {args.output}")

        if isinstance(result, list):
            # --no-merge returns one ReactionPathResult per shard.
            if not args.quiet:
                total = sum(len(r.paths) for r in result)
                print(f"Found {total} reaction paths in {len(result)} shard(s).")
                for r in result:
                    for species in r.auto_catalytic_species:
                        print(f"  {species}: {len(r.paths)} paths")
            if not args.output and not args.quiet:
                print("Use --output to save the shard results (pickle).")
        elif isinstance(result, ReactionPathResult):
            if not args.quiet:
                print(f"Found {len(result.paths)} reaction paths.")
                for i, idx in enumerate(result.index):
                    name = (result.total_reactions[i]
                            if i < len(result.total_reactions) else f"#{i}")
                    print(f"  {name}: {len(idx)} paths")
            if not args.output and not args.quiet:
                print("Use --output to save the paths (pickle).")
        elif "steady_coverage" in result:
            # CSTR result.
            if not args.quiet:
                print(f"CSTR finished: {result.get('message', '')}".rstrip())
                for name, value in zip(result.get("gas_names", []),
                                       result.get("steady_pressure", [])):
                    print(f"  p({name}) = {value:.6e}")
        else:
            results = result.get("results", [])
            if not args.quiet:
                print(f"Completed {len(results)} point(s).")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"catmkm: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
