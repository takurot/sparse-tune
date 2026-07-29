"""Command-line interface for the v0.1 public API."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import __version__
from ._benchmark import _environment, benchmark, list_backends
from ._inspect import diagnose_matrix
from ._profile import ProfileMismatchError, solve, tune
from ._types import BenchmarkResult, MatrixInfo, SolveStatus


_INPUT_ERROR = 2
_SOLVE_FAILED = 3


def _backend_ids(value: str) -> list[str]:
    aliases = {
        "scipy": "scipy:cpu",
        "cupy": "cupy:cuda:0",
    }
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one backend is required")
    return [aliases.get(item, item) for item in values]


def _measures(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or not set(values).issubset({"end-to-end", "steady-state"}):
        raise argparse.ArgumentTypeError(
            "measure must contain end-to-end or steady-state"
        )
    return values


def _add_common_benchmark_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backends", type=_backend_ids, default=_backend_ids("scipy,cupy")
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--measure",
        type=_measures,
        default=_measures("end-to-end,steady-state"),
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--tol", type=float, dest="rtol", help=argparse.SUPPRESS)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--assume-spd", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparsetune")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a matrix")
    inspect_parser.add_argument("matrix", type=Path)
    inspect_parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
    )
    inspect_parser.add_argument("--quiet", action="store_true")

    bench_parser = subparsers.add_parser("bench", help="benchmark backends")
    bench_parser.add_argument("matrix", type=Path)
    _add_common_benchmark_options(bench_parser)
    bench_parser.add_argument(
        "--format",
        choices=("json", "table", "csv"),
        default="json",
    )

    tune_parser = subparsers.add_parser("tune", help="save a tuning profile")
    tune_parser.add_argument("matrix", type=Path)
    _add_common_benchmark_options(tune_parser)
    tune_parser.add_argument("--output", type=Path, required=True)

    solve_parser = subparsers.add_parser("solve", help="solve with one backend")
    solve_parser.add_argument("matrix", type=Path)
    selection = solve_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile", type=Path)
    selection.add_argument("--backend")
    solve_parser.add_argument("--rhs", type=Path)
    solve_parser.add_argument(
        "--selection-mode",
        choices=("end-to-end", "steady-state"),
        default="end-to-end",
    )
    solve_parser.add_argument("--dtype", choices=("float32", "float64"))
    solve_parser.add_argument("--rtol", type=float)
    solve_parser.add_argument("--tol", type=float, dest="rtol", help=argparse.SUPPRESS)
    solve_parser.add_argument("--atol", type=float)
    solve_parser.add_argument("--max-iter", type=int)
    solve_parser.add_argument("--timeout", type=float, default=300.0)
    solve_parser.add_argument("--assume-spd", action="store_true")
    solve_parser.add_argument("--allow-stale-profile", action="store_true")
    solve_parser.add_argument(
        "--output",
        type=Path,
        help="write the solution vector as Matrix Market",
    )
    solve_parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress and metrics output",
    )

    doctor_parser = subparsers.add_parser("doctor", help="inspect the environment")
    doctor_parser.add_argument(
        "--format",
        choices=("json", "table"),
        default="table",
    )
    doctor_parser.add_argument("--quiet", action="store_true")
    return parser


def _progress(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr)


def _matrix_table(info: MatrixInfo) -> str:
    rows = [
        ("path", info.path),
        ("shape", f"{info.shape[0]} x {info.shape[1]}"),
        ("nnz", str(info.nnz)),
        ("density", f"{info.density:.6g}"),
        ("symmetry_ratio", f"{info.symmetry_ratio:.6g}"),
        ("diagonal_sign", info.diagonal_sign),
        ("spd_status", info.spd_status),
        ("fingerprint", info.fingerprint),
    ]
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"{name:<{width}}  {value}" for name, value in rows)


def _benchmark_payload(report: BenchmarkResult) -> dict[str, Any]:
    return report.to_dict()


def _benchmark_csv(report: BenchmarkResult) -> str:
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "backend",
            "status",
            "dtype",
            "total_seconds",
            "solve_seconds",
            "relative_residual",
            "error",
        )
    )
    for result in report.results:
        writer.writerow(
            (
                result.backend,
                result.status.value,
                result.dtype,
                result.total_seconds,
                result.solve_seconds,
                result.relative_residual,
                result.error or "",
            )
        )
    return output.getvalue().rstrip("\n")


def _benchmark_table(report: BenchmarkResult) -> str:
    header = f"{'backend':<20} {'status':<18} {'total (s)':>12} {'solve (s)':>12}"
    rows = [header]
    rows.extend(
        f"{result.backend:<20} {result.status.value:<18} "
        f"{result.total_seconds:>12.6g} {result.solve_seconds:>12.6g}"
        for result in report.results
    )
    for key, recommendation in report.recommendations.items():
        rows.append(
            f"{key}: {recommendation.backend or 'none'} - {recommendation.reason}"
        )
    return "\n".join(rows)


def environment_info() -> dict[str, Any]:
    """Return doctor information only when explicitly requested."""

    return {
        "version": __version__,
        **_environment(),
        "backends": list_backends(),
    }


def _environment_table(environment: dict[str, Any]) -> str:
    return "\n".join(
        f"{key}: {json.dumps(value) if isinstance(value, (list, dict)) else value}"
        for key, value in environment.items()
    )


def _run_inspect(args: argparse.Namespace) -> int:
    _progress(f"Inspecting {args.matrix}", quiet=args.quiet)
    info = diagnose_matrix(args.matrix)
    print(info.to_json() if args.format == "json" else _matrix_table(info))
    return 0


def _benchmark_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backends": args.backends,
        "dtype": args.dtype,
        "measure": args.measure,
        "runs": args.runs,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_iter": args.max_iter,
        "timeout": args.timeout,
        "assume_spd": args.assume_spd,
    }


def _run_bench(args: argparse.Namespace) -> int:
    _progress(f"Benchmarking {args.matrix}", quiet=args.quiet)
    report = benchmark(args.matrix, **_benchmark_options(args))
    if args.format == "json":
        print(json.dumps(_benchmark_payload(report)))
    elif args.format == "csv":
        print(_benchmark_csv(report))
    else:
        print(_benchmark_table(report))
    return 0


def _run_tune(args: argparse.Namespace) -> int:
    _progress(f"Tuning {args.matrix}", quiet=args.quiet)
    profile = tune(
        args.matrix,
        output=args.output,
        **_benchmark_options(args),
    )
    print(json.dumps(profile))
    return 0


def _run_solve(args: argparse.Namespace) -> int:
    _progress(f"Solving {args.matrix}", quiet=args.quiet)
    result = solve(
        args.matrix,
        rhs=args.rhs,
        profile=args.profile,
        backend=args.backend,
        selection_mode=args.selection_mode,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
        max_iter=args.max_iter,
        timeout=args.timeout,
        assume_spd=args.assume_spd,
        allow_stale_profile=args.allow_stale_profile,
        output=args.output,
    )
    if not args.quiet:
        print(result.to_metrics_json())
    return 0 if result.status is SolveStatus.CONVERGED else _SOLVE_FAILED


def _run_doctor(args: argparse.Namespace) -> int:
    _progress("Inspecting environment", quiet=args.quiet)
    environment = environment_info()
    print(
        json.dumps(environment)
        if args.format == "json"
        else _environment_table(environment)
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable process exit code."""

    args = _parser().parse_args(argv)
    commands = {
        "inspect": _run_inspect,
        "bench": _run_bench,
        "tune": _run_tune,
        "solve": _run_solve,
        "doctor": _run_doctor,
    }
    try:
        return commands[args.command](args)
    except (OSError, TypeError, ValueError, ProfileMismatchError) as error:
        print(f"error: {error}", file=sys.stderr)
        return _INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
