from pathlib import Path
import subprocess
import sys


def test_benchmark_cli_exposes_help() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--data-dir" in result.stdout
