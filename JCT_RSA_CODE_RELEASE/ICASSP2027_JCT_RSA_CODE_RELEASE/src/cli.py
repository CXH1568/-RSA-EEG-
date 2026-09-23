"""Command dispatch without import-time experiment execution."""

import argparse
import importlib
from pathlib import Path
import sys

from src.runtime import load_settings, release_root, resolve_path

WORKFLOWS = {
    'primary': 'ridge_clip_specificity_v2',
    'temporal': 'heldout_temporal_fixed_model',
    'controls': 'partial_rsa_variance_partition',
}


def main(kind, argv=None):
    if kind not in (*WORKFLOWS, 'figures'):
        raise ValueError('Unknown command: ' + kind)
    parser = argparse.ArgumentParser(
        description=('Render user-supplied specificity summaries without recomputing statistics.'
                     if kind == 'figures' else
                     'Run historical submission ' + kind + ' analysis on user-supplied data.'),
        epilog='No data or results are bundled. Help does not load scientific dependencies.',
    )
    parser.add_argument('--config', default=None, help='INI file; default ./configs/paths.ini')
    parser.add_argument('--data-root', help='Override DATA_ROOT')
    parser.add_argument('--images-root', help='Override IMAGES_ROOT')
    parser.add_argument('--output-root', help='Override RESULTS_ROOT')
    if kind == 'figures':
        parser.add_argument('--input', type=Path, required=True,
                            help='Directory containing your specificity temporal and heldout CSV summaries')
        parser.add_argument('--output', type=Path,
                            help='New PDF/PNG path; default RESULTS_ROOT/figures/specificity.pdf')
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        for option, key in (('data_root', 'DATA_ROOT'), ('images_root', 'IMAGES_ROOT'),
                            ('output_root', 'RESULTS_ROOT')):
            value = getattr(args, option)
            if value is not None:
                settings['paths'][key] = value
        if kind == 'figures':
            from src.visualization import render
            source = args.input if args.input.is_absolute() else release_root() / args.input
            output = args.output or resolve_path(settings, 'RESULTS_ROOT') / 'figures' / 'specificity.pdf'
            if not output.is_absolute():
                output = release_root() / output
            render(source, output)
        else:
            module = importlib.import_module('src.workflows.' + WORKFLOWS[kind])
            module.run(settings)
        return 0
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except (ValueError, KeyError) as exc:
        print('Invalid configuration or input: ' + str(exc), file=sys.stderr)
        return 2
    except ImportError as exc:
        print('Missing dependency. Install environment/requirements.txt. ' + str(exc), file=sys.stderr)
        return 4

