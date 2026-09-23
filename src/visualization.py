"""Specificity plot adapted from ridge_clip_specificity_v2 (saved summaries only)."""

from pathlib import Path


def render(input_path, output_path):
    """Render saved means/SEMs; never fit models or recompute group statistics.

    input_path is a directory containing clip_specificity_v2_temporal.csv and
    clip_specificity_v2_heldout.csv produced by the primary workflow.
    """
    folder = Path(input_path)
    temporal_path = folder / 'clip_specificity_v2_temporal.csv'
    heldout_path = folder / 'clip_specificity_v2_heldout.csv'
    for path in (temporal_path, heldout_path):
        if not path.is_file():
            raise FileNotFoundError('Result file not found: ' + path.name +
                                    '. Supply your own primary workflow outputs; none are bundled.')
    output = Path(output_path)
    if output.suffix.lower() not in ('.pdf', '.png'):
        raise ValueError('Figure output must be PDF or PNG.')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing figure: ' + output.name)

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    targets = ['CLIP', 'ResNet50', 'PixelPCA', 'Random']
    colors = {'CLIP': 'steelblue', 'ResNet50': 'forestgreen',
              'PixelPCA': 'darkorange', 'Random': 'gray'}
    temporal = pd.read_csv(temporal_path)
    heldout = pd.read_csv(heldout_path)
    needed = ['center_ms'] + [name + suffix for name in targets for suffix in ('_mean', '_sem')]
    if not set(needed) <= set(temporal.columns):
        raise ValueError('Temporal summary does not match specificity-v2 columns.')
    if not {'target', 'mean_r', 'sem_r', 'n_subjects'} <= set(heldout.columns):
        raise ValueError('Heldout summary does not match specificity-v2 columns.')
    if len(heldout) != 4 or heldout['target'].duplicated().any() or set(heldout['target']) != set(targets):
        raise ValueError('Expected exactly four distinct specificity targets.')
    heldout = heldout.set_index('target').loc[targets]
    numeric = temporal[needed].to_numpy(dtype=float)
    ho_numeric = heldout[['mean_r', 'sem_r', 'n_subjects']].to_numpy(dtype=float)
    if len(temporal) == 0 or not np.isfinite(numeric).all() or not np.isfinite(ho_numeric).all():
        raise ValueError('Input summary contains empty or non-finite values.')
    centers = temporal['center_ms'].to_numpy(dtype=float)
    if (np.diff(centers) <= 0).any():
        raise ValueError('Window centers must be strictly increasing.')
    for name in targets:
        if (temporal[name + '_sem'] < 0).any():
            raise ValueError('SEM cannot be negative.')
    if (heldout['sem_r'] < 0).any():
        raise ValueError('SEM cannot be negative.')
    counts = heldout['n_subjects'].to_numpy(dtype=float)
    if (counts < 2).any() or (counts != np.floor(counts)).any() or (counts != counts[0]).any():
        raise ValueError('Heldout target participant counts must agree and be valid integers.')

    # Original plot structure/styles. All displayed statistics are read, not computed.
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})
    try:
        ax = axes[0]
        for name in targets:
            mean_curve = temporal[name + '_mean'].to_numpy(dtype=float)
            sem_curve = temporal[name + '_sem'].to_numpy(dtype=float)
            ax.fill_between(centers, mean_curve - sem_curve, mean_curve + sem_curve,
                            alpha=0.15, color=colors[name])
            ax.plot(centers, mean_curve, color=colors[name], linewidth=2,
                    marker='o', markersize=3, label=name)
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8, label='Stimulus onset')
        ax.axvspan(-75, -25, alpha=0.08, color='blue', label='Strict pre')
        ax.axvspan(125, 300, alpha=0.08, color='red', label='Post target')
        ax.set_ylabel('RDM correlation (Spearman r)', fontsize=11)
        ax.set_title('Specificity v2: supplied temporal summaries', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax2 = axes[1]
        means = heldout['mean_r'].to_numpy(dtype=float)
        sems = heldout['sem_r'].to_numpy(dtype=float)
        bars = ax2.bar(targets, means, yerr=sems, capsize=5,
                       color=[colors[name] for name in targets], alpha=0.8, edgecolor='black')
        ax2.axhline(0, color='gray', linestyle=':', linewidth=0.7)
        ax2.set_ylabel('Held-out r', fontsize=11)
        ax2.set_title(f'Held-out at 175 ms (N={int(counts[0])})', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, mean, sem in zip(bars, means, sems):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 0.001,
                     f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
