import numpy as np
import plotly.graph_objects as go

from core.tube_method import (
    generate_middle_trajectory_method,
    generate_paper_like_trajectory_method,
    G,
)


# =========================================================
# Plot both methods together
# =========================================================
def plot_two_methods(
    target_point,
    release_point,
    middle_result,
    paper_result,
    show_middle_family=True,
    show_paper_family=True,
):
    fig = go.Figure()

    target_point = np.asarray(target_point, dtype=float)
    release_point = np.asarray(release_point, dtype=float)

    # -------------------------
    # Middle-method family
    # -------------------------
    if show_middle_family:
        for i, traj in enumerate(middle_result["family_trajs"]):
            if traj is None:
                continue
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode="lines",
                line=dict(color="royalblue", width=3),
                opacity=0.25,
                name="Middle method family" if i == 0 else None,
                showlegend=(i == 0),
            ))

    # Selected middle trajectory
    traj_mid = middle_result["selected_traj"]
    if traj_mid is not None:
        fig.add_trace(go.Scatter3d(
            x=traj_mid[:, 0], y=traj_mid[:, 1], z=traj_mid[:, 2],
            mode="lines",
            line=dict(color="blue", width=9),
            name="Middle method selected",
        ))

    # -------------------------
    # Paper-like family
    # -------------------------
    if show_paper_family:
        for i, traj in enumerate(paper_result["family_ball_trajs"]):
            if traj is None:
                continue
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode="lines",
                line=dict(color="crimson", width=3, dash="dash"),
                opacity=0.30,
                name="Paper-like family" if i == 0 else None,
                showlegend=(i == 0),
            ))

    # Selected paper-like trajectory
    traj_paper = paper_result["selected_ball_traj"]
    if traj_paper is not None:
        fig.add_trace(go.Scatter3d(
            x=traj_paper[:, 0], y=traj_paper[:, 1], z=traj_paper[:, 2],
            mode="lines",
            line=dict(color="red", width=9),
            name="Paper-like selected",
        ))

    # -------------------------
    # Common markers
    # -------------------------
    fig.add_trace(go.Scatter3d(
        x=[release_point[0]],
        y=[release_point[1]],
        z=[release_point[2]],
        mode="markers",
        marker=dict(color="black", size=7),
        name="Nominal release point",
    ))

    fig.add_trace(go.Scatter3d(
        x=[target_point[0]],
        y=[target_point[1]],
        z=[target_point[2]],
        mode="markers",
        marker=dict(color="green", size=9, symbol="diamond"),
        name="Target",
    ))

    # Ground plane
    xg = np.array([
        min(release_point[0], target_point[0]) - 0.3,
        max(release_point[0], target_point[0]) + 0.3,
        max(release_point[0], target_point[0]) + 0.3,
        min(release_point[0], target_point[0]) - 0.3
    ])
    yg = np.array([-1.0, -1.0, 1.0, 1.0])
    zg = np.array([target_point[2]] * 4)

    fig.add_trace(go.Mesh3d(
        x=xg, y=yg, z=zg,
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=0.10,
        color="gray",
        name="Ground",
        hoverinfo="skip",
    ))

    title = (
        "Comparison of two trajectory-generation methods"
        "<br><sup>"
        f"Middle method selected T = {middle_result['selected_time']:.3f} s; "
        f"Paper-like tube acceleration = {np.round(paper_result['tube_acceleration'], 3)}"
        "</sup>"
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=70),
    )

    fig.show()


# =========================================================
# Main demo
# =========================================================
if __name__ == "__main__":
    target_point = np.array([2.0, 0.0, 0.0])
    release_point = np.array([0.4, 0.0, 0.9])

    # Method 1: current middle-time method
    middle_result = generate_middle_trajectory_method(
        target_point=target_point,
        release_point=release_point,
        flight_time_range=(0.45, 0.85),
        n_family=21,
        g=G,
        n_pts=120,
    )

    # Method 2: simplified paper-like method
    paper_result = generate_paper_like_trajectory_method(
        target_point=target_point,
        release_point=release_point,
        nominal_flight_time=0.65,
        release_window=0.12,
        g=G,
        n_window_samples=13,
        n_ball_pts=120,
    )

    print("=== Method 1: Middle-time method ===")
    print("Selected flight time:", middle_result["selected_time"])
    print("Selected velocity   :", np.round(middle_result["selected_velocity"], 4))
    print()

    print("=== Method 2: Paper-like tube-acceleration method ===")
    print("Nominal velocity    :", np.round(paper_result["nominal_velocity"], 4))
    print("Tube acceleration   :", np.round(paper_result["tube_acceleration"], 4))
    print("Release window      :", paper_result["release_window"])
    print("Selected rel. point :", np.round(paper_result["selected_release_point"], 4))
    print("Selected rel. vel   :", np.round(paper_result["selected_release_velocity"], 4))

    # Set these to False if you only want the two highlighted trajectories
    plot_two_methods(
        target_point=target_point,
        release_point=release_point,
        middle_result=middle_result,
        paper_result=paper_result,
        show_middle_family=True,
        show_paper_family=True,
    )
