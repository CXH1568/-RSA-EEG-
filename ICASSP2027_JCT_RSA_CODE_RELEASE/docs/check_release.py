"""Limited release checks: no workflows, datasets, or render functions are run."""

import argparse
import ast
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ('run_primary', 'run_temporal', 'run_controls', 'generate_figures')
WORKFLOWS = (
    'fixed_model_subject_robustness', 'fixed_model_window_robustness',
    'heldout_temporal_fixed_model', 'heldout_uncertainty',
    'intermediate_layer_partial_rsa', 'layerwise_hierarchy_upgrade',
    'partial_rsa_variance_partition', 'ridge_clip_specificity_v2',
    'ridge_fixed_model_sweep', 'ridge_group_temporal_permutation_fullcohort',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('entries', 'workflows', 'all'), default='all')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    if args.stage == 'entries':
        for name in ENTRIES:
            assert (ROOT / 'scripts' / (name + '.py')).is_file(), 'Missing entry: ' + name
        print('ENTRY_FILES: PASS (4)')
        return
    for name in WORKFLOWS:
        path = ROOT / 'src' / 'workflows' / (name + '.py')
        assert path.is_file(), 'Missing workflow: ' + name
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
        # No executable top-level statements in workflow modules.
        for node in tree.body:
            assert isinstance(node, ast.FunctionDef) or (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ), 'Unsafe top-level statement: ' + name
        mod = importlib.import_module('src.workflows.' + name)
        assert callable(mod.run), name
    print('WORKFLOW_SYNTAX_IMPORT: PASS (10; run not called)')
    if args.stage == 'workflows':
        return
    paths = sorted(ROOT.rglob('*.py'))
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
        if path.resolve() == Path(__file__).resolve():
            continue
        name = '_release_check_' + '_'.join(path.relative_to(ROOT).with_suffix('').parts)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    for name in ENTRIES:
        result = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'scripts' / (name + '.py')), '--help'],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, name + ': ' + result.stderr
        assert '--config' in result.stdout, name
        print(name + ' --help: PASS')
    allowed = {'.py', '.ini', '.yml', '.yaml', '.md', '.txt'}
    for path in ROOT.rglob('*'):
        if path.is_dir():
            assert path.name not in {'data', 'results', 'checkpoints', 'logs', '__pycache__', '.git'}
        else:
            assert path.suffix.lower() in allowed or path.name in {'LICENSE', '.gitignore'}, path.name
    print('SYNTAX_IMPORT: PASS (' + str(len(paths)) + ' Python files)')
    print('FILE_ALLOWLIST: PASS')
    print('EXPERIMENTS: NOT_RUN')


if __name__ == '__main__':
    main()
