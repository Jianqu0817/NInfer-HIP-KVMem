#!/usr/bin/env python3
"""Compile and run standalone host tests; GPU integration tests are excluded."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", help="GNU-compatible C++ compiler executable")
    parser.add_argument("--std", default="c++17")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "build/host-tests")
    args = parser.parse_args()
    compiler = args.cxx or os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler and Path(r"D:\Dev-Cpp\TDM-GCC-64\bin\g++.exe").is_file():
        compiler = r"D:\Dev-Cpp\TDM-GCC-64\bin\g++.exe"
    if not compiler:
        print("ERROR: no host C++ compiler; pass --cxx <g++/clang++>", file=sys.stderr)
        return 1
    # Root tests have an explicit host suffix; tests/host owns any additional host-only cases.
    tests = set((ROOT / "tests").glob("*_state_test.cpp"))
    tests.update((ROOT / "tests/host").rglob("*.cpp"))
    tests = sorted(p for p in tests if not p.name.endswith("_gpu_test.cpp"))
    if not tests:
        print("ERROR: no standalone host tests found", file=sys.stderr)
        return 1
    out = args.out.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    compiler_bin = Path(compiler).resolve().parent
    local_paths = [compiler_bin]
    if (compiler_bin / "cpp").is_dir():
        local_paths.insert(0, compiler_bin / "cpp")
    env["PATH"] = os.pathsep.join(map(str, local_paths)) + os.pathsep + env.get("PATH", "")
    process_options = {}
    if os.name == "nt":
        # Child compilers inherit this mode; missing runtime DLLs return an error instead of
        # interrupting the desktop with a Windows critical-error dialog.
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        process_options["creationflags"] = 0x08000000
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                             errors="replace", env=env, timeout=30, **process_options)
    if version.returncode:
        print(f"ERROR: compiler did not run (exit {version.returncode})", file=sys.stderr)
        return 1
    receipt = {"compiler": compiler, "version": version.stdout.splitlines()[0],
               "standard": args.std, "compile_only": args.compile_only, "tests": []}
    failures = 0
    for source in tests:
        name = "_".join(source.relative_to(ROOT / "tests").with_suffix("").parts)
        exe = out / (name + (".exe" if os.name == "nt" else ""))
        command = [compiler, "-std=" + args.std, "-Wall", "-Wextra", "-pedantic",
                   "-I", str(ROOT / "src/infer/src"), str(source), "-o", str(exe)]
        result = subprocess.run(command, capture_output=True, env=env, timeout=120, **process_options)
        (out / (name + ".compile.log")).write_bytes(result.stdout + result.stderr)
        entry = {"source": str(source.relative_to(ROOT)), "command": command,
                 "compile_exit": result.returncode, "run_exit": None}
        if result.returncode:
            failures += 1
            print("COMPILE FAIL:", source.relative_to(ROOT))
        elif args.compile_only:
            print("COMPILED:", source.relative_to(ROOT))
        else:
            try:
                result = subprocess.run([str(exe)], capture_output=True, env=env, timeout=60,
                                        **process_options)
                output = result.stdout + result.stderr
                entry["run_exit"] = result.returncode
            except subprocess.TimeoutExpired as error:
                output = (error.stdout or b"") + (error.stderr or b"")
                entry["run_exit"] = "timeout"
            (out / (name + ".run.log")).write_bytes(output)
            if entry["run_exit"] != 0:
                failures += 1
            print(("PASS: " if entry["run_exit"] == 0 else "RUN FAIL: ") + str(source.relative_to(ROOT)))
            if output:
                print(output.decode("utf-8", errors="replace").rstrip())
        receipt["tests"].append(entry)
    (out / "results.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"{len(tests) - failures}/{len(tests)} passed; receipts: {out / 'results.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.SubprocessError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
