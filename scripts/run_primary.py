"""Command-line entry for primary; importing never starts an analysis."""

from pathlib import Path
import sys


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.cli import main as dispatch
    return dispatch('primary', argv)


if __name__ == '__main__':
    raise SystemExit(main())

