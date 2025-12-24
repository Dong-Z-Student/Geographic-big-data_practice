# -*- coding: utf-8 -*-
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def lifted_colormap(cmap, minval=0.3, maxval=1.0, n=256):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f"lifted({cmap.name})", new_colors
    )


# =========================
# 工具函数
# =========================

def split_categories(categories_str):
    if not categories_str:
        return []
    return [c.strip() for c in categories_str.split(',') if c.strip()]


# =========================
# 统计类别共现
# =========================

def build_category_cooccurrence(json_path, top_n=20):
    category_count = defaultdict(int)
    cooccur = defaultdict(lambda: defaultdict(int))

    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                continue

            cats = set(split_categories(data.get("categories", "")))
            if len(cats) < 2:
                continue

            for c in cats:
                category_count[c] += 1

            for a, b in itertools.combinations(cats, 2):
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    top_categories = sorted(
        category_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    categories = [c for c, _ in top_categories]
    node_sizes = np.array([category_count[c] for c in categories])

    n = len(categories)
    matrix = np.zeros((n, n))
    for i, a in enumerate(categories):
        for j, b in enumerate(categories):
            matrix[i, j] = cooccur[a].get(b, 0)

    return categories, node_sizes, matrix


# =========================
# 画弦图（论文级）
# =========================

def draw_chord_diagram(categories, node_sizes, matrix, out_png):
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    fig, ax = plt.subplots(
        figsize=(14, 18),
        subplot_kw=dict(polar=True)
    )
    ax.set_ylim(0, 11)
    ax.axis('off')

    # ---------- 颜色映射 ----------
    cmap = plt.cm.Blues
    node_norm = Normalize(vmin=node_sizes.min(), vmax=node_sizes.max())
    edge_norm = Normalize(vmin=matrix[matrix > 0].min(), vmax=matrix.max())

    # ---------- 画节点 ----------
    for i, (cat, angle) in enumerate(zip(categories, angles)):
        size = 400 + 2000 * node_norm(node_sizes[i])
        color = cmap(0.3 + 0.7 * node_norm(node_sizes[i]))

        ax.scatter(angle, 9.8, s=size, color=color, zorder=3)

        rotation = np.degrees(angle)
        align = 'left'
        if np.pi / 2 < angle < 3 * np.pi / 2:
            rotation += 180
            align = 'right'

        ax.text(
            angle,
            10.6,
            cat,
            rotation=rotation,
            ha=align,
            va='center',
            fontsize=10
        )

    # ---------- 画弦 ----------
    # ---------- 画弦（仅 Restaurants 与其他类别） ----------
    if "Restaurants" not in categories:
        print("WARNING: 'Restaurants' not in top categories, no edges drawn.")
    else:
        r_idx = categories.index("Restaurants")
        max_w = matrix[r_idx].max()

        for j in range(n):
            if j == r_idx:
                continue

            w = matrix[r_idx, j]
            if w == 0:
                continue

            lw = 0.5 + 6 * (w / max_w)
            color = cmap(0.3 + 0.7 * edge_norm(w))

            verts = [
                (angles[r_idx], 9.8),
                (angles[r_idx], 3.0),
                (angles[j], 3.0),
                (angles[j], 9.8)
            ]

            path = Path(verts, [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4
            ])

            patch = PathPatch(
                path,
                facecolor='none',
                edgecolor=color,
                lw=lw,
                alpha=0.6
            )
            ax.add_patch(patch)

    # ---------- 标题 ----------
    plt.title(
        "Category Co-occurrence Chord Diagram",
        fontsize=22,
        pad=70
    )

    # ---------- 图例 ----------
    cmap_lifted = lifted_colormap(plt.cm.Blues, 0.3, 1.0)
    sm_node = ScalarMappable(norm=node_norm, cmap=cmap)
    sm_node.set_array([])

    cbar = plt.colorbar(
        sm_node,
        ax=ax,
        fraction=0.03,
        pad=0.08
    )
    cbar.set_label("Category Frequency", fontsize=11)

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"弦图已保存：{out_png}")


# =========================
# main
# =========================

if __name__ == "__main__":
    json_file = r"data/yelp_academic_dataset_business.json"

    categories, node_sizes, matrix = build_category_cooccurrence(
        json_file,
        top_n=40
    )

    draw_chord_diagram(
        categories,
        node_sizes,
        matrix,
        out_png="类别共现弦图.png"
    )
