"""
Created on 2022-01-13

description: hazard analysis - plot landfall intensity histograms (from track or windfield data)

@author: Thomas Vogt (tvogts)
"""

import pathlib
import sys

import matplotlib.gridspec
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ['IBTrACS', 'IBTrACS_p', 'STORM', 'MIT', 'CHAZ_ERA5']
DATASET_NAMES = ['IBTrACS', 'IBTrACS_p', 'STORM', 'MIT', 'CHAZ']

REGIONS = ['AP', 'IO', 'SH', 'WP']
REGIONS_LONG = {
    'AP': 'North Atlantic/Eastern Pacific (AP)',
    'IO': 'North Indian Ocean (IO)',
    'SH': 'Southern Hemisphere (SH)',
    'WP': 'Western Pacific (WP)',
}

CATEGORIES = ['Trop. storm', 'Cat. 1', 'Cat. 2', 'Cat. 3', 'Cat. 4', 'Cat. 5']
N_CATEGORIES = len(CATEGORIES)

COLORS = ['k', '#3e3f40', '#b7222b', '#1116cb', '#2abdc7', 'g']

DATA_DIR = pathlib.Path("./data")
STATS_DIR = DATA_DIR / "stats"
OUTPUT_DIR = DATA_DIR / "plots" / "histograms"


def ax_make_bare(ax):
    for sp in ['left', 'right', 'top', 'bottom']:
        ax.spines[sp].set_visible(False)


def plot_stats_separate_ds(ax, dataset, stats, bar_width):
    i_dataset = DATASETS.index(dataset)
    for i_region, region in enumerate(REGIONS):
        t_data = stats[dataset][region]['fq']
        ax.bar(i_region + bar_width * np.arange(N_CATEGORIES), t_data,
               color=COLORS[i_dataset], width=0.8 * bar_width)
    ax.set_xlim(left=-bar_width, right=len(REGIONS) - 2 * bar_width)
    ax.yaxis.set_ticks([])
    # ax_make_bare(ax)


def plot_stats_separate(subplot_spec, stats):
    bar_width = 0.8 / N_CATEGORIES

    outer = matplotlib.gridspec.GridSpecFromSubplotSpec(
        2, 1, hspace=0, height_ratios=(100, 23), subplot_spec=subplot_spec)

    ax_legend = plt.subplot(outer[1])
    ax_legend.legend(
        loc=(-0.05, -0.1), fontsize=14, borderaxespad=0,
        ncol=3, columnspacing=2.0,
        handles=[
            matplotlib.patches.Patch(color=color, label=name)
            for name, color in zip(DATASET_NAMES, COLORS)
        ]) #.get_frame().set_linewidth(0)
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])
    ax_make_bare(ax_legend)

    inner = matplotlib.gridspec.GridSpecFromSubplotSpec(
        len(DATASETS), 1, hspace=0, subplot_spec=outer[0])

    axs = []
    for i_dataset, dataset in enumerate(DATASETS):
        ax = plt.subplot(inner[i_dataset], sharex=axs[-1] if len(axs) > 0 else None)
        plot_stats_separate_ds(ax, dataset, stats, bar_width)
        axs.append(ax)

    ymax = axs[0].get_ylim()[1]
    for i_region, region in enumerate(REGIONS):
        axs[0].text(i_region + 1.2 * bar_width, 1.05 * ymax, region, fontsize=14)
    axs[0].text(-0.01, 1.0, "e)", transform=axs[0].transAxes,
                fontsize=14, va='bottom', ha='right')


    n_regions = len(REGIONS)
    axs[-1].set_xticks((
        np.arange(n_regions)[:, None] + bar_width * np.arange(N_CATEGORIES)[None]
    ).ravel())
    axs[-1].set_xticklabels(CATEGORIES * n_regions, rotation=90)

    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)


def plot_stats_mingled_reg(ax0, ax1, reg, stats, is_left, shared_ax2=None):
    x_pos = np.array(range(len(CATEGORIES)))
    bar_width = 0.25
    m_reg = REGIONS.index(reg)

    # create boxplots
    bp_pos = -1
    for i, (name, color) in enumerate(zip(DATASETS, COLORS)):
        if name == "IBTrACS":
            continue
        bp = ax1.boxplot(
            stats[name][reg]['freq_normed'],
            positions=x_pos * 2.0 + bp_pos * bar_width,
            widths=bar_width - 0.04,
            flierprops=dict(marker='o',
                            markerfacecolor='none',
                            markersize=3,
                            markeredgecolor=color))
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

        bp_pos += 1

    # plot histograms
    ax2 = ax0.twinx()
    ax2.set_yscale('log')
    if shared_ax2 is not None:
        ax2.get_shared_y_axes().join(ax2, shared_ax2)
    hist_pos = -2
    for i, (name, color) in enumerate(zip(DATASETS, COLORS)):
        ax0.bar(
            x_pos[:5] * 2.0 + hist_pos * bar_width,
            stats[name][reg]['fq'][:5],
            yerr=stats[name][reg]['std'][:5],
            width=bar_width, color=color, alpha=0.8)
        ax2.bar(
            x_pos[-1:] * 2.0 + hist_pos * bar_width,
            stats[name][reg]['fq'][-1:],
            yerr=stats[name][reg]['std'][-1:],
            width=bar_width, color=color, alpha=0.8)
        hist_pos += 1

    ax0.axvline(9, lw=0.5, color='0.5', linestyle='--')
    ax0.set_title(REGIONS_LONG[reg], y=1.0, pad=4, fontsize=14)
    ax0.text(-0.03, 1.0, f"{'abcd'[m_reg]})", transform=ax0.transAxes,
             fontsize=14, va='bottom', ha='right')

    ax1.axhline(1, lw=0.5, color='0.5', linestyle='--')
    ymax = ax1.get_ylim()[1]
    if ymax < 3.9:
        ymax = 3.9
        ax1.set_yticks([0, 1, 2, 3])
    ax1.set_ylim(bottom=-0.05 * ymax, top=ymax)
    ax1.set_yticks([t for t in ax1.get_yticks() if t < 0.93 * ymax and t >= 0])
    ax1.set_xticks(range(0, N_CATEGORIES * 2, 2))
    ax1.set_xticklabels(CATEGORIES)

    if is_left:
        label_coords = (-0.1, 0.5)
        ax0.set_ylabel('probability density', fontsize=14)
        ax0.get_yaxis().set_label_coords(*label_coords)
        ax1.set_ylabel('rel. variability', fontsize=14)
        ax1.get_yaxis().set_label_coords(*label_coords)

    return ax2


def plot_stats_mingled(subplot_spec, stats):
    outer = matplotlib.gridspec.GridSpecFromSubplotSpec(
        2, 2, wspace=0.15, hspace=0.1, subplot_spec=subplot_spec)
    ax0_base = None
    ax1_base = None
    ax2_base = None
    for m, reg in enumerate(REGIONS):
        inner = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 1, height_ratios=[2, 1], hspace=0, subplot_spec=outer[m])
        ax0 = plt.subplot(inner[0], sharex=ax0_base, sharey=ax0_base)
        ax1 = plt.subplot(inner[1], sharex=ax1_base)
        ax2 = plot_stats_mingled_reg(ax0, ax1, reg, stats, m in [0, 2], shared_ax2=ax2_base)
        if m in [0, 1]:
            plt.setp(ax1.get_xticklabels(), visible=False)
        if m in [0, 2]:
            plt.setp(ax2.get_yticklabels(), visible=False)
        if m in [1, 3]:
            plt.setp(ax0.get_yticklabels(), visible=False)
        if m == 0:
            ax0_base = ax0
            ax1_base = ax1
            ax2_base = ax2
    ax0_base.set_ylim(bottom=0)
    ymax = ax0_base.get_ylim()[1]
    ax0_base.set_yticks([t for t in ax0.get_yticks() if t < 0.97 * ymax and t >= 0])
    ax2_base.set_ylim(bottom=1e-6)


def plot_stats(path, stats, with_separate):
    fig = plt.figure(figsize=(16, 9))

    if with_separate:
        outer = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        plot_stats_separate(outer[1], stats)
    else:
        outer = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[20, 1])
        ax_legend = plt.subplot(outer[1])
        ax_legend.legend(
            loc="center", fontsize=14, borderaxespad=0,
            ncol=len(DATASET_NAMES), columnspacing=2.0,
            handles=[
                matplotlib.patches.Patch(color=color, label=name)
                for name, color in zip(DATASET_NAMES, COLORS)
            ]) #.get_frame().set_linewidth(0)
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        ax_make_bare(ax_legend)
    plot_stats_mingled(outer[0], stats)

    print(f"Writing to {path} ...")
    fig.tight_layout()
    fig.savefig(path, dpi=600, facecolor='w',
                edgecolor='w', orientation='portrait',
                format='pdf', bbox_inches='tight', pad_inches=0.1)


def main(mode=None, with_separate=False):
    suffix = "_wsep" if with_separate else ""

    if mode is None or mode.lower() in ["haz", "hazard", "hazards"]:
        plot_stats(OUTPUT_DIR / f'from_hazard{suffix}.pdf', get_haz_stats(), with_separate)
        return

    for thresh in [0, 50, 100, 150, 200, 250, 300]:
        for mode in ["max", "all"]:
            plot_stats(
                OUTPUT_DIR / f"from_tracks_{mode}_{thresh:d}{suffix}.pdf",
                get_track_stats(mode, thresh),
                with_separate)


def norm_freq(stats):
    fq_sum = stats['fq'].sum()
    stats['fq'] /= fq_sum
    stats['std'] /= fq_sum
    return stats


def get_haz_stats():
    stats = {}
    for tr in DATASETS:
        stats[tr] = {}
        for reg in REGIONS:
            path = STATS_DIR / f"haz_hist_{reg}_{tr}.npz"
            try:
                stats[tr][reg] = norm_freq(dict(np.load(path)))
            except FileNotFoundError:
                print(f"File not found: {path}")
                stats[tr][reg] = {
                    'fq': np.zeros(6),
                    'std': np.zeros(6),
                    'freq_normed': np.zeros((1, 6)),
                }
    return stats


def get_track_stats(mode, thresh):
    stats = {}
    for tr in DATASETS:
        stats[tr] = {}
        for reg in REGIONS:
            path = STATS_DIR / f"track_hist_{reg}_{tr}.npz"
            try:
                data = np.load(path)

                # remove NaN from freq_normed (from division by 0)
                freq_normed = data[f'freq_normed_{mode}_{thresh:d}']
                freq_normed[np.isnan(freq_normed)] = 0

                stats[tr][reg] = norm_freq({
                    'fq': data[f'fq_{mode}_{thresh:d}'],
                    'std': data[f'std_{mode}_{thresh:d}'],
                    'freq_normed': freq_normed,
                })
            except FileNotFoundError:
                print(f"File not found: {path}")
                stats[tr][reg] = {
                    'fq': np.zeros(6),
                    'std': np.zeros(6),
                    'freq_normed': np.zeros((1, 6)),
                }
    return stats


if __name__ == "__main__":
    main(mode=sys.argv[1] if len(sys.argv) > 1 else None,
         with_separate=len(sys.argv) > 2)
