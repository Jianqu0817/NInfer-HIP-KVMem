#!/usr/bin/env python3
"""Build the checked-in text Engine and CLI on Windows/HIP, without legacy objects."""
import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
INFER = ROOT / "src" / "infer"
SOURCE = INFER / "src"
PROCESS_OPTIONS = {}
if os.name == "nt":
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    PROCESS_OPTIONS["creationflags"] = 0x08000000


def sources():
    files = []
    for directory in ("core", "artifact", "text", "ops", "models", "runtime"):
        files.extend(p for p in (SOURCE / directory).rglob("*")
                     if p.suffix in (".cpp", ".cu"))
    files.append(SOURCE / "media" / "decode" / "decode_stub.cpp")
    files.append(INFER / "third_party" / "utf8proc" / "utf8proc.c")
    files.extend((INFER / "third_party" / "llama-jinja" / "jinja").glob("*.cpp"))
    return sorted(files)


def find_rocm(explicit):
    if explicit:
        return Path(explicit).resolve()
    for name in ("HIP_PATH", "ROCM_PATH"):
        if os.environ.get(name):
            return Path(os.environ[name]).resolve()
    hipcc = shutil.which("hipcc.exe")
    if hipcc:
        return Path(hipcc).resolve().parents[1]
    installs = sorted(Path(r"C:\Program Files\AMD\ROCm").glob("*"), reverse=True)
    return next((p for p in installs if (p / "bin" / "hipcc.exe").is_file()), None)


def find_vcvars(explicit):
    if explicit:
        return Path(explicit).resolve()
    installer = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vswhere = installer / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        result = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                 "-property", "installationPath"], capture_output=True, text=True,
                                check=True, **PROCESS_OPTIONS)
        if result.stdout.strip():
            return Path(result.stdout.strip()) / "VC/Auxiliary/Build/vcvars64.bat"
    return None


def build_environment(vcvars, rocm, out):
    env = dict(os.environ)
    if vcvars:
        bat = out / "vc_environment.bat"
        bat.write_text(f'@echo off\ncall "{vcvars}" >nul\nif errorlevel 1 exit /b 1\nset\n',
                       encoding="utf-8")
        result = subprocess.run(["cmd.exe", "/d", "/c", str(bat)], capture_output=True,
                                text=True, errors="replace", check=True, **PROCESS_OPTIONS)
        env.update(line.split("=", 1) for line in result.stdout.splitlines()
                   if "=" in line and not line.startswith("="))
    path_key = next((k for k in env if k.upper() == "PATH"), "PATH")
    tool_paths = [rocm / "bin"]
    llvm_bin = rocm / "lib/llvm/bin"
    if llvm_bin.is_dir():
        tool_paths.append(llvm_bin)
        env["HIP_CLANG_PATH"] = str(llvm_bin)
    device_libs = rocm / "lib/llvm/amdgcn/bitcode"
    if device_libs.is_dir():
        env["HIP_DEVICE_LIB_PATH"] = str(device_libs)
    env["HIP_PATH"] = str(rocm)
    env["ROCM_PATH"] = str(rocm)
    env[path_key] = os.pathsep.join(map(str, tool_paths)) + os.pathsep + env.get(path_key, "")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="inspect dependencies; no compilation")
    action.add_argument("--plan", action="store_true", help="list exact source manifest as JSON")
    action.add_argument("--build", action="store_true", help="compile, archive, then link")
    action.add_argument("--link-only", action="store_true", help="compile entry and link existing engine archive")
    parser.add_argument("--rocm", help="HIP SDK root containing bin/hipcc.exe")
    parser.add_argument("--vcvars", help="path to VS 2022 vcvars64.bat")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "windows")
    parser.add_argument("--jobs", type=int, default=2, help="parallel compilers (default 2 for 32 GB RAM)")
    parser.add_argument("--entry", type=Path, default=ROOT / "harness/mini_main.cpp",
                        help="C++ executable entry (relative paths resolve from repository root)")
    parser.add_argument("--name", default="ninfer_mini_nordc", help="executable basename, without .exe")
    parser.add_argument("--only", type=Path, action="append", default=[],
                        help="compile only this repository-relative source; repeatable, no archive/link")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not args.name or Path(args.name).name != args.name or args.name.endswith(".exe"):
        parser.error("--name must be a basename without .exe")
    if args.only and not args.build:
        parser.error("--only requires --build")
    entry = (ROOT / args.entry).resolve()
    files = sources()
    if args.plan:
        print(json.dumps({"root": str(ROOT), "sources": [str(p.relative_to(ROOT)) for p in files],
                          "entry": str(entry), "name": args.name}, indent=2))
        return 0
    rocm = find_rocm(args.rocm)
    vcvars = find_vcvars(args.vcvars)
    ar = None if rocm is None else next((p for p in
        (rocm / "bin/llvm-ar.exe", rocm / "lib/llvm/bin/llvm-ar.exe") if p.is_file()), None)
    missing = []
    for rel in ("bin/hipcc.exe", "include/hip/hip_runtime.h",
                "lib/hipblas.lib", "lib/rocblas.lib"):
        if rocm is None or not (rocm / rel).is_file():
            missing.append("HIP SDK: " + rel)
    if ar is None:
        missing.append("HIP SDK: bin/llvm-ar.exe or lib/llvm/bin/llvm-ar.exe")
    if vcvars is None and not shutil.which("cl.exe"):
        missing.append("MSVC 2022 C++ x64 toolchain (vcvars64.bat / cl.exe)")
    elif vcvars is not None and not vcvars.is_file():
        missing.append("vcvars64.bat: " + str(vcvars))
    missing.extend("source: " + str(p) for p in files if not p.is_file())
    if not entry.is_file():
        missing.append("entry: " + str(entry))
    print("Repository:", ROOT)
    print("Build process:", os.getpid())
    print("HIP SDK:", rocm or "not found")
    print("MSVC environment:", vcvars or shutil.which("cl.exe") or "not found")
    print("Engine translation units:", len(files))
    for item in missing:
        print("MISSING:", item)
    if missing:
        return 1
    if args.check:
        return 0
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    library = out / "ninfer_hip.lib"
    if args.link_only and not library.is_file():
        raise RuntimeError("--link-only requires an engine archive from a successful --build: " + str(library))
    env = build_environment(vcvars, rocm, out)
    env_path = next(v for k, v in env.items() if k.upper() == "PATH")
    cl = shutil.which("cl.exe", path=env_path)
    if not cl:
        raise RuntimeError("MSVC environment did not provide cl.exe")
    includes = [ROOT / "src/shim", SOURCE, INFER / "include", INFER / "third_party",
                INFER / "third_party/cub", INFER / "third_party/ggml-quants/include",
                INFER / "third_party/ggml-quants/src", INFER / "third_party/llama-jinja",
                INFER / "third_party/spdlog/include", rocm / "include"]
    hipcc = str(rocm / "bin/hipcc.exe")
    hip_flags = ["--offload-arch=gfx1201", "-mno-wavefrontsize64", "-std=c++20", "-O2",
                 "-D__HIP_PLATFORM_AMD__", "-D__CUDACC__", "-DGGML_USE_HIP", "-DRDNA4",
                 "-DNINFER_HW_WMMMA", "-DNINFER_K_TILE", "-DNINFER_ACCEPT_FORCE_SINGLE",
                 "-DNINFER_VEC_DEBUG_OFF", "-DUTF8PROC_STATIC"]
    device_libs = rocm / "lib/llvm/amdgcn/bitcode"
    device_flags = (["--rocm-device-lib-path=" + str(device_libs)]
                    if (device_libs / "ocml.bc").is_file() else [])
    hip_flags += device_flags
    host_flags = ["/O2", "/EHsc", "/utf-8", "/D__HIP_PLATFORM_AMD__",
                  "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX", "/DUTF8PROC_STATIC"]
    commands = []
    objects = []
    for source in files + [entry]:
        name = ("entry_" + args.name + ".obj" if source == entry else
                "_".join(source.relative_to(ROOT).parts) + ".obj")
        obj = out / name
        objects.append(obj)
        # gguf_linear shares HIP bfloat16 ABI with the device bridge.
        if source.suffix == ".cu" or source.name == "gguf_linear.cpp":
            command = [hipcc] + hip_flags + ["-I" + str(p) for p in includes]
            command += ["-c", str(source), "-o", str(obj)]
        else:
            language = ["/TC"] if source.suffix == ".c" else ["/std:c++20", "/Zc:__cplusplus"]
            command = [cl] + host_flags + language + ["/I" + str(p) for p in includes]
            command += ["/c", str(source), "/Fo" + str(obj)]
        commands.append((name, command))
    (out / "compile_commands.json").write_text(json.dumps(
        [{"directory": str(ROOT), "arguments": command, "file": str(source)}
         for source, (_, command) in zip(files + [entry], commands)],
        indent=2), encoding="utf-8")

    def run(job):
        name, command = job
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, **PROCESS_OPTIONS)
        (logs / (name + ".log")).write_bytes(result.stdout + result.stderr)
        (logs / (name + ".exitcode")).write_text(str(result.returncode), encoding="ascii")
        print(("OK " if result.returncode == 0 else "FAIL ") + name, flush=True)
        return result.returncode

    compile_jobs = commands[-1:] if args.link_only else commands
    if args.only:
        selected = {(ROOT / p).resolve() for p in args.only}
        available = files + [entry]
        unknown = selected.difference(available)
        if unknown:
            parser.error("--only source is not in the build manifest: " + str(next(iter(unknown))))
        compile_jobs = [job for source, job in zip(available, commands) if source in selected]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        failures = sum(code != 0 for code in executor.map(run, compile_jobs))
    if failures:
        print(f"{failures} compilation(s) failed; no archive or executable produced. Logs: {logs}")
        return 1
    if args.only:
        print(f"Compiled {len(compile_jobs)} selected unit(s); archive and executable unchanged.")
        return 0
    # Only this invocation's manifest contributes objects; never glob stale probe/legacy objects.
    if not args.link_only:
        library.unlink(missing_ok=True)
        archive_rsp = out / "archive.rsp"
        archive_rsp.write_text("\n".join('"' + str(p).replace("\\", "/") + '"' for p in objects[:-1]),
                               encoding="utf-8")
        if run(("archive", [str(ar), "rcs", str(library), "@" + str(archive_rsp)])):
            return 1
    exe = out / (args.name + ".exe")
    link_command = [hipcc, "--offload-arch=gfx1201", "-mno-wavefrontsize64", "-std=c++20", "-O2",
                    str(objects[-1]), str(library), str(rocm / "lib/hipblas.lib"),
                    str(rocm / "lib/rocblas.lib"), "-o", str(exe)]
    link_command += device_flags
    if run(("link", link_command)):
        return 1
    print("Built:", exe)
    print("Runtime DLL path (for this shell):", rocm / "bin")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
