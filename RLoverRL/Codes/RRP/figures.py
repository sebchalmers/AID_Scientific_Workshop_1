import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patches

# Keep the same TeX / minimalist aesthetic you use elsewhere
plt.rcParams['text.usetex'] = True
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_aspect('equal', 'box')

def _save(fig, name):
    fig.savefig(name, transparent=True, dpi='figure',
                bbox_inches='tight', pad_inches=0.05)
    print(f"saved: {os.path.abspath(name)}")

# -----------------------------------------------------------------------------
# Scenario tree figures (0/1/2)
# -----------------------------------------------------------------------------
def scenario_tree_fig(stage_depth=1, fname="ScenarioTree0.pdf",
                      node_r=0.06, lw=1.6):
    """
    Draw a scenario tree similar to your TikZ mock, with 3 stages maximum.
    stage_depth in {1,2,3} controls how many levels are shown.
    """
    # Layout coordinates (hand-tuned for a clean look)
    x0, y0 = 0.0, 0.0   # root
    x1 = 1.2            # stage 1
    x2 = 2.4            # stage 2
    x3 = 3.6            # stage 3
    ys1 = [+1.5, -1.5]
    ys2 = [+3.0, +1.0, -1.0, -3.0]
    ys3 = [+3.5, +2.5, +1.5, +0.5, -0.5, -1.5, -2.5, -3.5]

    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    _clean_axes(ax)

    # Root node
    ax.add_patch(plt.Circle((x0, y0), node_r, edgecolor='black',
                            facecolor='white', lw=lw))
    # Label root (small, consistent with your style)
    ax.text(x0 - 0.1, y0 - 0.3, r'$\mathbf{s},\,\mathbf{u}_0$',
            ha='right', va='top', fontsize=10)

    # Stage 1 nodes + edges
    if stage_depth >= 1:
        for y in ys1:
            ax.plot([x0, x1], [y0, y], color='black', lw=lw)
            ax.add_patch(plt.Circle((x1, y), node_r*0.75, edgecolor='black',
                                    facecolor='black', lw=lw))

    # Stage 2 nodes + edges
    if stage_depth >= 2:
        # connect each stage-1 node to two stage-2 nodes
        pairs = [(ys1[0], [ys2[0], ys2[1]]),
                 (ys1[1], [ys2[2], ys2[3]])]
        for y_from, y_to_list in pairs:
            for y_to in y_to_list:
                ax.plot([x1, x2], [y_from, y_to], color='black', lw=lw)
                ax.add_patch(plt.Circle((x2, y_to), node_r*0.7, edgecolor='black',
                                        facecolor='black', lw=lw))

    # Stage 3 nodes + edges
    if stage_depth >= 3:
        # connect each stage-2 node to two stage-3 nodes
        pairs2 = [(ys2[0], [ys3[0], ys3[1]]),
                  (ys2[1], [ys3[2], ys3[3]]),
                  (ys2[2], [ys3[4], ys3[5]]),
                  (ys2[3], [ys3[6], ys3[7]])]
        for y_from, y_to_list in pairs2:
            for y_to in y_to_list:
                ax.plot([x2, x3], [y_from, y_to], color='black', lw=lw)
                ax.add_patch(plt.Circle((x3, y_to), node_r*0.65, edgecolor='black',
                                        facecolor='black', lw=lw))

    # Light caption at the bottom (optional)
    if stage_depth == 1:
        ax.text(0.5*(x0+x1), -4.2, r'\scriptsize First branching',
                ha='center', va='top')
    elif stage_depth == 2:
        ax.text(0.5*(x1+x2), -4.2, r'\scriptsize Second stage branching',
                ha='center', va='top')
    else:
        ax.text(0.5*(x1+x3), -4.2, r'\scriptsize Scenario tree (3 stages)',
                ha='center', va='top')

    # Fix limits to keep layout stable
    ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(-4.2, 4.2)

    _save(fig, fname)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Rolling-horizon (robust MPC) figures
# -----------------------------------------------------------------------------
def rolling_horizon_fig(T=18, H=6, fname="RollingHorizon0.pdf",
                        lw=1.6, tube_alpha=0.18):
    """
    Show a time axis with a nominal trajectory and a receding horizon window.
    Also draw a thin "tube" conveying uncertainty across scenarios (like your containers).
    """
    # Nominal trend + small wiggles (just for aesthetics)
    t = np.arange(T+1)
    y_nom = 0.7*np.exp(-0.05*t) * np.cos(0.3*t)

    # Safe band and tightened constraints band
    y_safe_lo, y_safe_hi = -0.8, 0.8
    y_tight_lo, y_tight_hi = -0.55, 0.55

    fig = plt.figure(figsize=(7.5, 2.8))
    ax = fig.add_subplot(111)

    # Safe band (light gray)
    ax.axhspan(y_safe_lo, y_safe_hi, color='0.85', alpha=0.6, zorder=0)
    # Tightened constraints band (green)
    ax.axhspan(y_tight_lo, y_tight_hi, color='#7fc97f', alpha=0.35, zorder=1)

    # Across-scenarios spread (epistemic tube)
    spread = 0.25*np.exp(-0.06*t)  # shrinking tube (replanning helps)
    y_up = y_nom + spread
    y_dn = y_nom - spread

    # Tube
    ax.fill_between(t, y_dn, y_up, color='C0', alpha=tube_alpha, linewidth=0, zorder=2)
    # Nominal line
    ax.plot(t, y_nom, color='C0', lw=lw, zorder=3)

    # Text labels on right
    ax.text(T+0.2, 0.5*(y_safe_lo+y_safe_hi), 'Safe set', fontsize=8, va='center', color='black')
    ax.text(T+0.2, 0.5*(y_tight_lo+y_tight_hi), 'Tightened (robust) constraints', fontsize=8, va='center', color='black')
    ax.text(T+0.2, y_nom[-1], 'Uncertainty tube', fontsize=8, va='center', color='C0')

    # Arrow annotation from green band to blue tube
    arrow_y = 0.0
    ax.annotate('guaranteed containment',
                xy=(T*0.7, y_tight_hi), xycoords='data',
                xytext=(T*0.7, y_tight_hi + 0.3), textcoords='data',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
                fontsize=8, color='green', ha='center')

    # First horizon box
    ax.axvspan(0, H, color='C3', alpha=0.12)
    ax.text(0.5*H, max(y_up)+0.02, r'\scriptsize plan horizon', ha='center', va='bottom')

    # "Solve" marker at t=0
    ax.plot([0], [y_nom[0]], marker='o', color='k', ms=4)
    ax.text(0, min(y_dn)-0.16, r'\scriptsize solve MPC', ha='center', va='top')

    # A little annotation showing "apply u0"
    ax.annotate(r'\scriptsize apply $u_0$', xy=(0, y_nom[0]), xytext=(1.2, y_nom[0]+0.25),
                arrowprops=dict(arrowstyle='->', lw=1), fontsize=8)

    # Minimal axis look
    ax.set_xlim(-0.5, T+0.5)
    # Compact y-lims with a margin
    ax.set_ylim(min(y_dn)-0.2, max(y_up)+0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ["top","right","left","bottom"]:
        ax.spines[s].set_visible(False)

    _save(fig, fname)
    plt.close(fig)

def rolling_horizon_shifted(T=18, H=6, shift=3, fname="RollingHorizon1.pdf",
                            lw=1.6, tube_alpha=0.18):
    """
    Same plot, but with the horizon shifted right by `shift` to illustrate re-solve.
    """
    t = np.arange(T+1)
    y_nom = 0.7*np.exp(-0.05*t) * np.cos(0.3*t)

    # Safe band and tightened constraints band
    y_safe_lo, y_safe_hi = -0.8, 0.8
    y_tight_lo, y_tight_hi = -0.55, 0.55

    spread = 0.25*np.exp(-0.06*t)
    y_up = y_nom + spread
    y_dn = y_nom - spread

    fig = plt.figure(figsize=(7.5, 2.8))
    ax = fig.add_subplot(111)

    # Safe band (light gray)
    ax.axhspan(y_safe_lo, y_safe_hi, color='0.85', alpha=0.6, zorder=0)
    # Tightened constraints band (green)
    ax.axhspan(y_tight_lo, y_tight_hi, color='#7fc97f', alpha=0.35, zorder=1)

    # Tube
    ax.fill_between(t, y_dn, y_up, color='C0', alpha=tube_alpha, linewidth=0, zorder=2)
    ax.plot(t, y_nom, color='C0', lw=lw, zorder=3)

    # Text labels on right
    ax.text(T+0.2, 0.5*(y_safe_lo+y_safe_hi), 'Safe set', fontsize=8, va='center', color='black')
    ax.text(T+0.2, 0.5*(y_tight_lo+y_tight_hi), 'Tightened (robust) constraints', fontsize=8, va='center', color='black')
    ax.text(T+0.2, y_nom[-1], 'Uncertainty tube', fontsize=8, va='center', color='C0')

    # Arrow annotation from green band to blue tube at shifted solve point
    ax.annotate('guaranteed containment',
                xy=(shift+H*0.5, y_tight_hi), xycoords='data',
                xytext=(shift+H*0.5, y_tight_hi + 0.3), textcoords='data',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
                fontsize=8, color='green', ha='center')

    # Shifted horizon
    ax.axvspan(shift, shift+H, color='C3', alpha=0.12)
    ax.text(shift + 0.5*H, max(y_up)+0.02, r'\scriptsize re-plan horizon',
            ha='center', va='bottom')

    # "Re-solve" marker at t=shift
    ax.plot([shift], [y_nom[shift]], marker='o', color='k', ms=4)
    ax.text(shift, min(y_dn)-0.16, r'\scriptsize re-solve MPC', ha='center', va='top')

    # "apply u0" from the new solve
    ax.annotate(r'\scriptsize apply $u_0$', xy=(shift, y_nom[shift]),
                xytext=(shift+1.2, y_nom[shift]+0.25),
                arrowprops=dict(arrowstyle='->', lw=1), fontsize=8)

    ax.set_xlim(-0.5, T+0.5)
    ax.set_ylim(min(y_dn)-0.2, max(y_up)+0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ["top","right","left","bottom"]:
        ax.spines[s].set_visible(False)

    _save(fig, fname)
    plt.close(fig)

# -----------------------------------------------------------------------------
# New robust scenario tree figure
# -----------------------------------------------------------------------------
def scenario_tree_robust_fig(fname="ScenarioTree_Robust.pdf", node_r=0.06, lw=1.6):
    """
    Draw a 3-stage scenario tree with robust feasibility annotations:
    - leaf nodes show constraint status (green check or red X)
    - root annotated with a translucent green box labeled 'intersection of scenario constraints'
    - dashed lines connect this box to stage-1 branches
    """
    # Layout coordinates (same as scenario_tree_fig stage_depth=3)
    x0, y0 = 0.0, 0.0   # root
    x1 = 1.2            # stage 1
    x2 = 2.4            # stage 2
    x3 = 3.6            # stage 3
    ys1 = [+1.5, -1.5]
    ys2 = [+3.0, +1.0, -1.0, -3.0]
    ys3 = [+3.5, +2.5, +1.5, +0.5, -0.5, -1.5, -2.5, -3.5]

    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    _clean_axes(ax)

    # Root node
    ax.add_patch(plt.Circle((x0, y0), node_r, edgecolor='black',
                            facecolor='white', lw=lw))
    ax.text(x0 - 0.1, y0 - 0.3, r'$\mathbf{s},\,\mathbf{u}_0$',
            ha='right', va='top', fontsize=10)

    # Stage 1 nodes + edges
    for y in ys1:
        ax.plot([x0, x1], [y0, y], color='black', lw=lw)
        ax.add_patch(plt.Circle((x1, y), node_r*0.75, edgecolor='black',
                                facecolor='black', lw=lw))

    # Stage 2 nodes + edges
    pairs = [(ys1[0], [ys2[0], ys2[1]]),
             (ys1[1], [ys2[2], ys2[3]])]
    for y_from, y_to_list in pairs:
        for y_to in y_to_list:
            ax.plot([x1, x2], [y_from, y_to], color='black', lw=lw)
            ax.add_patch(plt.Circle((x2, y_to), node_r*0.7, edgecolor='black',
                                    facecolor='black', lw=lw))

    # Stage 3 nodes + edges
    pairs2 = [(ys2[0], [ys3[0], ys3[1]]),
              (ys2[1], [ys3[2], ys3[3]]),
              (ys2[2], [ys3[4], ys3[5]]),
              (ys2[3], [ys3[6], ys3[7]])]
    for y_from, y_to_list in pairs2:
        for y_to in y_to_list:
            ax.plot([x2, x3], [y_from, y_to], color='black', lw=lw)
            ax.add_patch(plt.Circle((x3, y_to), node_r*0.65, edgecolor='black',
                                    facecolor='black', lw=lw))

    # Leaf nodes constraint bars and status marks
    # Draw small vertical bars at each leaf node to represent constraints
    bar_height = 0.3
    bar_width = 0.02
    # Select which leaves violate constraints (red X) and which are feasible (green check)
    violating_indices = [1, 5]  # example indices in ys3 list (0-based)
    # All others are feasible
    for i, y_leaf in enumerate(ys3):
        # vertical bar
        rect = Rectangle((x3 - bar_width/2, y_leaf - bar_height/2),
                         bar_width, bar_height,
                         facecolor='lightgray', edgecolor='black', lw=1)
        ax.add_patch(rect)
        # mark status
        if i in violating_indices:
            ax.text(x3 + 0.1, y_leaf, r'$\times$', color='#d95f02', fontsize=14, va='center', ha='left')
        else:
            ax.text(x3 + 0.1, y_leaf, r'$\surd$', color='#1b9e77', fontsize=12, va='center', ha='left')

    # Draw a light green translucent rectangle near the root labeled 'intersection of scenario constraints'
    rect_width = 0.9
    rect_height = 1.2
    rect_x = x0 - rect_width/2
    rect_y = y0 - rect_height/2
    robust_rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                            facecolor='#7fc97f', alpha=0.25, edgecolor='none', zorder=0)
    ax.add_patch(robust_rect)
    ax.text(x0, y0, 'intersection of\nscenario constraints',
            fontsize=8, color='black', ha='center', va='center')

    # Connect the rectangle with thin dashed lines to the two stage-1 branches
    for y in ys1:
        ax.plot([x0 + rect_width/2, x1], [y0, y], color='black', lw=0.8, linestyle='dashed')

    # Bottom caption
    ax.text(0.5*(x1+x3), -4.2, r'\scriptsize Robust stochastic program: enforce constraints on every scenario (min-max).',
            ha='center', va='top')

    # Fix limits to keep layout stable
    ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(-4.2, 4.2)

    _save(fig, fname)
    plt.close(fig)

def scenario_tree_chance_fig(fname="ScenarioTree_Chance.pdf", node_r=0.06, lw=1.6):
    """
    Draw a 3-stage scenario tree with chance constraint feasibility annotations:
    - leaf nodes show mostly green checkmarks and a few red Xs
    - no green intersection rectangle at root
    - caption at bottom explaining chance constraints
    """
    # Layout coordinates (same as scenario_tree_fig stage_depth=3)
    x0, y0 = 0.0, 0.0   # root
    x1 = 1.2            # stage 1
    x2 = 2.4            # stage 2
    x3 = 3.6            # stage 3
    ys1 = [+1.5, -1.5]
    ys2 = [+3.0, +1.0, -1.0, -3.0]
    ys3 = [+3.5, +2.5, +1.5, +0.5, -0.5, -1.5, -2.5, -3.5]

    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    _clean_axes(ax)

    # Root node
    ax.add_patch(plt.Circle((x0, y0), node_r, edgecolor='black',
                            facecolor='white', lw=lw))
    ax.text(x0 - 0.1, y0 - 0.3, r'$\mathbf{s},\,\mathbf{u}_0$',
            ha='right', va='top', fontsize=10)

    # Stage 1 nodes + edges
    for y in ys1:
        ax.plot([x0, x1], [y0, y], color='black', lw=lw)
        ax.add_patch(plt.Circle((x1, y), node_r*0.75, edgecolor='black',
                                facecolor='black', lw=lw))

    # Stage 2 nodes + edges
    pairs = [(ys1[0], [ys2[0], ys2[1]]),
             (ys1[1], [ys2[2], ys2[3]])]
    for y_from, y_to_list in pairs:
        for y_to in y_to_list:
            ax.plot([x1, x2], [y_from, y_to], color='black', lw=lw)
            ax.add_patch(plt.Circle((x2, y_to), node_r*0.7, edgecolor='black',
                                    facecolor='black', lw=lw))

    # Stage 3 nodes + edges
    pairs2 = [(ys2[0], [ys3[0], ys3[1]]),
              (ys2[1], [ys3[2], ys3[3]]),
              (ys2[2], [ys3[4], ys3[5]]),
              (ys2[3], [ys3[6], ys3[7]])]
    for y_from, y_to_list in pairs2:
        for y_to in y_to_list:
            ax.plot([x2, x3], [y_from, y_to], color='black', lw=lw)
            ax.add_patch(plt.Circle((x3, y_to), node_r*0.65, edgecolor='black',
                                    facecolor='black', lw=lw))

    # Leaf nodes constraint bars and status marks
    bar_height = 0.3
    bar_width = 0.02
    # Select which leaves violate constraints (red X) and which are feasible (green check)
    # For chance constraints, most are feasible, a few violate
    violating_indices = [3, 6]  # example indices in ys3 list (0-based)
    for i, y_leaf in enumerate(ys3):
        rect = Rectangle((x3 - bar_width/2, y_leaf - bar_height/2),
                         bar_width, bar_height,
                         facecolor='lightgray', edgecolor='black', lw=1)
        ax.add_patch(rect)
        if i in violating_indices:
            ax.text(x3 + 0.1, y_leaf, r'$\times$', color='#d95f02', fontsize=14, va='center', ha='left')
        else:
            ax.text(x3 + 0.1, y_leaf, r'$\surd$', color='#1b9e77', fontsize=12, va='center', ha='left')

    # Bottom caption
    ax.text(0.5*(x1+x3), -4.2, r'\scriptsize Chance constraint: allow small violation probability in scenarios.',
            ha='center', va='top')

    # Fix limits to keep layout stable
    ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(-4.2, 4.2)

    _save(fig, fname)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Scenario trees
    scenario_tree_fig(stage_depth=1, fname="ScenarioTree0.pdf")
    scenario_tree_fig(stage_depth=2, fname="ScenarioTree1.pdf")
    scenario_tree_fig(stage_depth=3, fname="ScenarioTree2.pdf")

    # Rolling horizon (robust MPC)
    rolling_horizon_fig(T=18, H=6, fname="RollingHorizon0.pdf")
    rolling_horizon_shifted(T=18, H=6, shift=3, fname="RollingHorizon1.pdf")

    # Robust scenario tree figure
    scenario_tree_robust_fig()

    # Chance constraint scenario tree figure
    scenario_tree_chance_fig()