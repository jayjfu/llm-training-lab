import os
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

MAX_OUTPUT_CHARS = 20_000
LAUNCHER = """
import os, resource, sys
memory_mb, cpu_seconds, file_mb = map(int, sys.argv[1:4])
if memory_mb > 0:
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
resource.setrlimit(resource.RLIMIT_FSIZE, (file_mb << 20, file_mb << 20))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.execv(sys.argv[4], sys.argv[4:])
"""


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


@cache
def network_prefix():
    # unprivileged user + network namespace: the child sees no network interfaces
    unshare = shutil.which("unshare")
    if unshare and subprocess.run([unshare, "-rn", "true"], capture_output=True).returncode == 0:
        return [unshare, "-rn"]
    print("warning: `unshare -rn` unavailable, sandboxed code keeps network access", file=sys.stderr)
    return []


def run_command(command, cwd, timeout, memory_mb=2048, file_mb=64, extra_env=None, isolate_network=True):
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(cwd), "TMPDIR": str(cwd), "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        **(extra_env or {}),
    }
    limits = [sys.executable, "-I", "-c", LAUNCHER, str(memory_mb), str(int(timeout) + 1), str(file_mb)]
    full_command = (network_prefix() if isolate_network else []) + limits + [shutil.which(command[0]) or command[0], *command[1:]]
    process = subprocess.Popen(full_command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, errors="replace", start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        timed_out = True
    return RunResult(process.returncode, stdout[-MAX_OUTPUT_CHARS:], stderr[-MAX_OUTPUT_CHARS:], timed_out)


def parse_junit(path):
    if not path.exists():
        return []
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError:
        return []
    return [not any(child.tag in ("failure", "error", "skipped") for child in case) for case in root.iter("testcase")]


@lru_cache(maxsize=4096)
def run_tests(code, test_source, timeout=10.0, memory_mb=2048):
    """Runs a pytest file against `solution.py` and returns one bool per test (empty on timeout or no code)."""
    if not code.strip():
        return ()
    with tempfile.TemporaryDirectory(prefix="sandbox_") as work_dir:
        work_path = Path(work_dir)
        (work_path / "solution.py").write_text(code)
        (work_path / "test_solution.py").write_text(test_source)
        report_path = work_path / "report.xml"
        command = [sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(report_path), "test_solution.py"]
        result = run_command(command, work_path, timeout, memory_mb)
        return () if result.timed_out else tuple(parse_junit(report_path))


def run_tests_batch(codes, test_sources, timeout=10.0, memory_mb=2048, max_workers=None):
    max_workers = max_workers or min(32, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda pair: run_tests(*pair, timeout=timeout, memory_mb=memory_mb), zip(codes, test_sources)))