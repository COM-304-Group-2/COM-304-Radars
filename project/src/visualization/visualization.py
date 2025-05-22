

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
    ax.set_ylim(0, 150)

    # self.ax_3.set_aspect('equal', adjustable='box')  # keep units equal
    ax.autoscale(enable=False)

    for tr in tracks:
        if tr['status'] != 'ACTIVE':
            continue

        x, y = tr['pos']
        vx, vy = tr['vel']
        uid = tr['uid']
        confidence = tr.get('confidence', 1.0)

        # draw the position
        ax.scatter(x, y, s=100 * confidence, edgecolors='k', facecolors='none')
        # draw an arrow showing velocity
        ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, width=0.005)

        # label with the track ID
        ax.text(x, y, f"{uid}", fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.5))

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("GTRACK 2D Tracks (size ∝ confidence)")
    ax.grid(True)