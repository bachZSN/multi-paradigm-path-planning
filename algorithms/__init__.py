"""Algorithms package.

Keep this module lightweight: importing torch-based diffusion models here
would make the app slow to start. Diffusion components are exposed via
lazy attribute access.
"""

from .astar import astar, dijkstra_search, bfs, calculate_total_cost, reconstruct_path

__all__ = [
    "astar",
    "dijkstra_search",
    "bfs",
    "calculate_total_cost",
    "reconstruct_path",
    # diffusion symbols are available lazily via __getattr__
    "PathUNet",
    "DDPM",
    "ddim_sample_cfg",
    "infer_path",
    "TrajectoryUNet1D",
    "TrajectoryDDPM",
    "infer_path_coord",
]


def __getattr__(name: str):
    if name in {"PathUNet", "DDPM", "ddim_sample_cfg", "infer_path"}:
        from .diffusion import PathUNet, DDPM, ddim_sample_cfg, infer_path
        return {"PathUNet": PathUNet, "DDPM": DDPM, "ddim_sample_cfg": ddim_sample_cfg, "infer_path": infer_path}[name]
    if name in {"TrajectoryUNet1D", "TrajectoryDDPM", "infer_path_coord"}:
        from .diffusion_coord import TrajectoryUNet1D, TrajectoryDDPM, infer_path_coord
        return {
            "TrajectoryUNet1D": TrajectoryUNet1D,
            "TrajectoryDDPM": TrajectoryDDPM,
            "infer_path_coord": infer_path_coord,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
