"""Run untrusted, model-written Python code and report whether it passed.

Two modes:
    process (default)  separate Python process + timeout + throwaway folder + CPU/memory limits.
                       Stops accidents (infinite loops, crashes, memory blow-ups, stray files).
                       NOT a security boundary: the code can still read your files and use the network.
    docker             the same program inside a sealed container: no network, read-only file
                       system, memory/CPU/process limits, thrown away afterwards.
                       Use it with:  SANDBOX=docker uv run python router/stage1_baseline.py ...

A program "passes" when it exits with code 0: every assert in the tests held.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

MEMORY_MB = 512
DOCKER_IMAGE = "python:3.12-slim"


def _limit_resources(cpu_seconds):
    """Runs inside the child process just before the code starts (process mode, Linux/macOS)."""
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))      # CPU time
    try:                                                                       # memory: not enforced
        mem = MEMORY_MB * 1024 * 1024                                          # on every macOS version
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except (ValueError, OSError):
        pass
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))       # no files over 10 MB


def run_code(code: str, timeout: float = 10.0, mode: str | None = None) -> dict:
    """Run `code` as its own program. Returns {passed, timed_out, error}."""
    mode = mode or os.environ.get("SANDBOX", "process")
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "program.py").write_text(code)

        if mode == "docker":
            if shutil.which("docker") is None:
                raise RuntimeError("SANDBOX=docker but the docker command isn't installed")
            name = f"sandbox-{uuid.uuid4().hex[:12]}"
            cmd = ["docker", "run", "--rm", "--name", name,
                   "--network", "none",                     # no internet
                   "--memory", f"{MEMORY_MB}m", "--cpus", "1",
                   "--pids-limit", "64",                    # no fork bombs
                   "--read-only", "--tmpfs", "/tmp",        # can't write anywhere but /tmp
                   "-v", f"{folder}:/code:ro",              # the program, read-only
                   DOCKER_IMAGE, "python", "-I", "/code/program.py"]
            kwargs = {}
        else:
            cmd = [sys.executable, "-I", "program.py"]       # -I: ignore env vars and local modules
            kwargs = dict(cwd=folder,
                          env={"PATH": "/usr/bin:/bin"},     # don't pass API keys to model code
                          preexec_fn=lambda: _limit_resources(int(timeout) + 1))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
        except subprocess.TimeoutExpired:
            if mode == "docker":                             # stopping docker run doesn't stop
                subprocess.run(["docker", "kill", name], capture_output=True)   # the container
            return dict(passed=False, timed_out=True, error=f"timed out after {timeout}s")

    return dict(passed=result.returncode == 0, timed_out=False,
                error="" if result.returncode == 0 else result.stderr[-500:])


if __name__ == "__main__":
    tests = {
        "passes":        "def add(a, b): return a + b\nassert add(2, 3) == 5",
        "wrong answer":  "def add(a, b): return a - b\nassert add(2, 3) == 5",
        "infinite loop": "while True: pass",
        "syntax error":  "def add(a, b) return a + b",
        "crashes":       "print(1 / 0)",
        "reads API key": "import os\nassert 'GROQ_API_KEY' not in os.environ",
        "huge memory":   "x = bytearray(4 * 1024**3)",
    }
    for name, code in tests.items():
        r = run_code(code, timeout=2)
        print(f"{name:14s} passed={r['passed']!s:5s} timed_out={r['timed_out']!s:5s} "
              f"{r['error'].strip().splitlines()[-1] if r['error'] else ''}")
