import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def configure_ax_bf(ax, phi, r, vmin=0, vmax=0.1):
    """
    Configure axes for beamforming visualization.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure.
    phi : numpy.ndarray
        Array of angles in radians.
    r : numpy.ndarray
        Array of radial distances.
    vmin : float, optional
        Minimum value for color scaling (default is 0).
    vmax : float, optional
        Maximum value for color scaling (default is 0.1).

    Returns:
    ----------
    im : matplotlib.collections.QuadMesh
        The pcolormesh object for the beamforming visualization.
    """

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_title("Bird Eye View (Top View)")

    R, Theta = np.meshgrid(r, phi)

    ax.set_xlim(phi[0], phi[-1])
    ax.set_ylim(r[0], r[-1])
    ax.grid(False)

    num_phi = len(phi)
    data = np.zeros((num_phi, r.shape[0]), dtype=np.float64)

    im = ax.pcolormesh(Theta, R, data, shading='nearest', cmap='jet', vmin=vmin, vmax=vmax)

    return im


def configure_ax_db(ax):
    """
    Configure axes for DBSCAN clustering visualization.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure.
    """

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("DBSCAN Clustering on Full Heatmap")


def configure_ax_gtrack(ax, width, rgd):
    """
    Configure axes for GTRACK visualization.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure.
    width : float
        The width of the area to visualize.
    rgd : float
        The range of the y-axis (height) for the visualization.
    """

    ax.set_xlim(-width, width)
    ax.set_ylim(0, rgd)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("GTRACK")
    ax.grid(True)


def update_ax_gtrack(ax, tracks, last_artists):
    """
    Update the GTRACK visualization axes with the current tracks.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axes to update.
    tracks : list of dict
        List of track dictionaries containing 'pos', 'vel', 'uid', and 'status'.
    last_artists : list
    """

    for art in last_artists:
        art.remove()
    last_artists.clear()
    # Filter active tracks
    active = [tr for tr in tracks if tr['status'] == 'ACTIVE']

    # Sort by UID
    ids = sorted({tr['uid'] for tr in active})

    # Defined colormap
    TRACK_COLORS = {}
    PALETTE = cm.get_cmap('Set2')

    def get_color_for_uid(uid):
        if uid not in TRACK_COLORS:
            # assign next slot in the palette
            next_idx = len(TRACK_COLORS) % PALETTE.N
            TRACK_COLORS[uid] = PALETTE(next_idx)
        return TRACK_COLORS[uid]

    # Draw each track
    for tr in tracks:
        x, y        = tr['pos']
        vx, vy      = tr['vel']
        uid         = tr['uid']
        col         = get_color_for_uid(uid)
        col_edge    = col

        if tr['status'] != 'ACTIVE':
            col = 'None'

        # Draw circle for each track
        sc = ax.scatter(x, y,
                   s=500,
                   facecolor=col,
                   edgecolor=col_edge,
                   linewidth=3,
                   zorder=3)

        # Draw arrow for each track
        qv = ax.quiver(x, y, vx, vy,
                  angles='xy',
                  scale_units='xy',
                  scale=0.3,
                  width=0.005,
                  color=col)

        last_artists.extend([sc, qv])

    # Build legend
    handles = [
        mpatches.Patch(color=get_color_for_uid(uid), label=str(uid))
        for uid in ids
    ]
    leg = ax.legend(handles=handles,
              title='Track ID',
              loc='center left',
              bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0)

    last_artists.append(leg)
