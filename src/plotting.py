import numpy as np
import matplotlib.pyplot as plt


def plot_disk(squares, boundary_coords, square_scale=1.0, highlight_square=None, show_grid=False, 
              show_square_centers=False, show_title=False, save_path=None):
    """
    Plot a square-lattice disk. 

    Visual parameters:
    square_scale: 
        Side length used when drawing each square. The default is 1.0, which
        matches the original lattice geometry.
    highlight_square: 
        Index of a square to highlight, for example the chosen start square.
    show_grid: 
        Whether to draw the lattice grid in the background.
    show_square_centers:
        Whether to plot the square-center points.
    show_title:
        Wheter to print a plot title.
    save_path:
        If provided, the plot is saved to this file path.
    """
    
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")

    x_coords = [square[0] for square in squares]
    y_coords = [square[1] for square in squares]

    x_min = int(np.floor(min(x_coords))) - 1
    x_max = int(np.ceil(max(x_coords))) + 2
    y_min = int(np.floor(min(y_coords))) - 1
    y_max = int(np.ceil(max(y_coords))) + 2

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if show_grid:
        ax.set_xticks(range(x_min, x_max))
        ax.set_yticks(range(y_min, y_max))
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Draw lattice squares
    for i, (x, y, neighbors) in enumerate(squares):
        if i == highlight_square:
            facecolor = "red"
            alpha = 0.4
        else:
            facecolor = "C0"
            alpha = 0.3

        ax.add_patch(
            plt.Rectangle(
                (x - 0.5 * square_scale, y - 0.5 * square_scale),
                square_scale,
                square_scale,
                facecolor=facecolor,
                edgecolor="none",
                alpha=alpha,
            )
        )

    # Optionally plot square centers
    center_size = 100/len(squares)
    
    if show_square_centers:
        ax.scatter(
            x_coords,
            y_coords,
            s=center_size,
            color="black",
            zorder=3
        )


    # Draw boundary
    for coords in boundary_coords:
        ax.plot(coords[:, 0], coords[:, 1], color="black", linewidth=0.6)
    
    if show_title:
        ax.set_title(
            f"Square-lattice disk: {len(squares)} squares, {len(boundary_coords)} boundary edges"
        )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

  def plot_disk_map_comparison(z_values, u_values, u_fit, w=None, boundary_mids=None, u_boundary_fit=None, save_path=None):
    """
    Plot a comparison of the discrete disk map in two panels:

    Left panel:
        - interior points z_i in the unit disk
        - boundary preimages w_j on the unit circle

    Right panel:
        - interior lattice points u_i
        - interpolated inverse map values \\tilde{F}(z_i)
        - optionally true lattice boundary midpoints
        - optionally approximated boundary images \\tilde{F}(w_j)
        
    Additional parameters:
    boundary_mids : True lattice boundary midpoints.
    u_boundary_fit : Approximated boundary images under the fitted inverse map.
    save_path : If provided, the plot is saved to this file path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left panel: unit disk
    # -----------------------------
    ax = axes[0]
    ax.set_aspect("equal")

    # Draw unit circle
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)

    # Interior disk points z_i
    ax.scatter(
        np.real(z_values),
        np.imag(z_values),
        s=20,
        label=r"pre-images $z_i$"
    )

    # Boundary preimages w_j
    if w is not None:
        ax.scatter(
            np.real(w),
            np.imag(w),
            s=20,
            label=r"pre-images $w_i$"
        )

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.legend(loc='upper right')

    # Add small margin around unit disk
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    
    # Right panel: lattice disk
    # -----------------------------
    ax = axes[1]
    ax.set_aspect("equal")

    # True interior lattice points
    ax.scatter(
        np.real(u_values),
        np.imag(u_values),
        s=35,
        marker="x",
        c='black',
        label=r"lattice points $u_i$"
    )

    # Interpolated inverse map on interior points
    ax.scatter(
        np.real(u_fit),
        np.imag(u_fit),
        s=20,
        label=r"$\tilde{F}(z_i)$"
    )

    # True boundary lattice points
    if boundary_mids is not None:
        ax.scatter(
            boundary_mids[:, 0],
            boundary_mids[:, 1],
            s=35,
            marker="x",
            c='black',
        )

    # Approximated boundary images
    if u_boundary_fit is not None:
        ax.scatter(
            np.real(u_boundary_fit),
            np.imag(u_boundary_fit),
            s=20,
            label=r"$\tilde{F}(w_i)$"
        )

    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.legend(loc='upper right')

    x_all = [np.real(u_values), np.real(u_fit)]
    y_all = [np.imag(u_values), np.imag(u_fit)]

    if boundary_mids is not None:
        x_all.append(boundary_mids[:, 0])
        y_all.append(boundary_mids[:, 1])

    if u_boundary_fit is not None:
        x_all.append(np.real(u_boundary_fit))
        y_all.append(np.imag(u_boundary_fit))

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)

    ax.set_xlim(x_all.min() - 0.5, x_all.max() + 0.5)
    ax.set_ylim(y_all.min() - 0.5, y_all.max() + 0.5)

    # Draw lattice grid
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(True, which='major', color='lightgray', linewidth=0.5)
    
    ax.grid(True, which='both', color='lightgray', linewidth=0.5)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
