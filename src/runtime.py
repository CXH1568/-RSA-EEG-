"""Portable paths and fail-fast input checks. No scientific calculations."""

import configparser
import csv
from pathlib import Path

DATASET_ERROR = (
    'Dataset not found.\n'
    'Please download the required dataset and configure DATA_ROOT.'
)


def release_root():
    return Path(__file__).resolve().parents[1]


def load_settings(config_path=None):
    path = Path(config_path) if config_path else release_root() / 'configs' / 'paths.ini'
    if not path.is_absolute():
        path = release_root() / path
    settings = configparser.ConfigParser(interpolation=None)
    with path.open(encoding='utf-8') as handle:
        settings.read_file(handle)
    for key in ('DATA_ROOT', 'RESULTS_ROOT', 'CONFIG_ROOT', 'IMAGES_ROOT'):
        if not settings.get('paths', key, fallback='').strip():
            raise ValueError('Missing configuration: ' + key)
    return settings


def resolve_path(settings, key):
    path = Path(settings['paths'][key]).expanduser()
    return (path if path.is_absolute() else release_root() / path).resolve()


def image_paths(images_root, concept, max_imgs=5):
    root = Path(images_root).resolve()
    folder = (root / str(concept)).resolve()
    if not folder.is_relative_to(root):
        raise ValueError('Concept image directory must be inside IMAGES_ROOT.')
    if not folder.is_dir():
        raise FileNotFoundError(DATASET_ERROR + '\nMissing concept image directory: ' + str(concept))
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files:
        raise FileNotFoundError(DATASET_ERROR + '\nMissing images for concept: ' + str(concept))
    return [str(p) for p in files[:max_imgs]]


def _brainvision_links(header):
    # Header/marker metadata only; no EEG values are opened here.
    try:
        text = header.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        text = header.read_text(encoding='cp1252')
    links = {}
    for line in text.splitlines():
        key, sep, value = line.partition('=')
        if sep and key.strip() in ('DataFile', 'MarkerFile'):
            value = value.strip().replace('\\', '/')
            linked = (header.parent / value).resolve()
            if not linked.is_file():
                raise FileNotFoundError(DATASET_ERROR + '\nMissing BrainVision dependency: ' + value)
            links[key.strip()] = linked
    return links


def require_dataset(settings):
    dataset = resolve_path(settings, 'DATA_ROOT')
    images = resolve_path(settings, 'IMAGES_ROOT')
    if not dataset.is_dir() or not images.is_dir():
        raise FileNotFoundError(DATASET_ERROR)
    concepts = set()
    for number in range(1, 51):
        subject = f'sub-{number:02d}'
        folder = dataset / subject / 'eeg'
        header = folder / f'{subject}_task-rsvp_eeg.vhdr'
        events = folder / f'{subject}_task-rsvp_events.tsv'
        if not header.is_file() or not events.is_file():
            raise FileNotFoundError(DATASET_ERROR + '\nMissing EEG header/events for ' + subject)
        links = _brainvision_links(header)
        if not {'DataFile', 'MarkerFile'} <= links.keys():
            raise ValueError('BrainVision header lacks DataFile/MarkerFile: ' + subject)
        _brainvision_links(links['MarkerFile'])
        with events.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if not {'istarget', 'onset', 'objectnumber', 'object'} <= set(reader.fieldnames or ()):
                raise ValueError('Missing RSVP event columns for ' + subject)
            for row in reader:
                if float(row['istarget']) == 0:
                    if not row['object']:
                        raise ValueError('Empty concept name for ' + subject)
                    concepts.add(row['object'])
    if not concepts:
        raise ValueError('No non-target concepts found in RSVP metadata.')
    for concept in sorted(concepts):
        image_paths(images, concept)
    return dataset
