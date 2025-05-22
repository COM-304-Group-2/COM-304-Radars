import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as mpatches

def configure_ax_bf(ax):
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_title("Bird Eye View (Top View)")


def configure_ax_db(ax):
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("DBSCAN Clustering on Full Heatmap")


def configure_ax_gtrack(ax, tracks):
    ax.set_xlim(-70, 70)
    ax.set_ylim(0, 100)
    #ax.autoscale(enable=False)

    # filter only active tracks
    active = [tr for tr in tracks if tr['status'] == 'ACTIVE']

    # decide what you want to group by: here I'm using the track 'uid' as a "cluster ID";
    # if you really have a separate tr['cluster'] field, swap out 'uid' for 'cluster' below
    ids = sorted({tr['uid'] for tr in active})

    # global (or module‐level) map:
    TRACK_COLORS = {}
    # pick whatever categorical palette you like
    PALETTE = cm.get_cmap('tab10')

    def get_color_for_uid(uid):
        if uid not in TRACK_COLORS:
            # assign next slot in the palette
            next_idx = len(TRACK_COLORS) % PALETTE.N
            TRACK_COLORS[uid] = PALETTE(next_idx)
        return TRACK_COLORS[uid]

    for tr in tracks:
        x, y        = tr['pos']
        vx, vy      = tr['vel']
        uid         = tr['uid']
        col         = get_color_for_uid(uid)
        col_edge    = col

        if tr['status'] != 'ACTIVE':
            col = 'None'

        # small filled circle with a black edge
        ax.scatter(x, y,
                   s=500,             # small marker size
                   facecolor=col,
                   edgecolor=col_edge,
                   linewidth=3,
                   zorder=3)

        # velocity arrow in same color
        ax.quiver(x, y, vx, vy,
                  angles='xy',
                  scale_units='xy',
                  scale=1,
                  width=0.005,
                  color=col)

    # build legend handles and place it to the right of the plot
    handles = [
        mpatches.Patch(color=get_color_for_uid(uid), label=str(uid))
        for uid in ids
    ]
    ax.legend(handles=handles,
              title='Track ID',
              loc='center left',
              bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("GTRACK 2D Tracks (color → cluster/ID)")
    ax.grid(True)


def plot_2d_heatmap(ax, data, theta, r, vmin=0, vmax=0.1):
    R, Theta = np.meshgrid(r, theta)
    ax.pcolormesh(Theta, R, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_xlim(theta[0], theta[-1])
    ax.set_ylim(r[0], r[-1])
    ax.grid(False)