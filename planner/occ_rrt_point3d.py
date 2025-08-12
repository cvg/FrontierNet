import numpy as np
from typing import Dict, List, Optional, Sequence, Union
from scipy.spatial import KDTree

from utils.planner_utils import (
    pose2posquat, posquat2pose, interpolate_waypoints
)
from utils.geometry import compute_alignment_transforms, pose_difference

# ompl pybind
from ompl import base as ob
from ompl import geometric as og


class OccupancyGrid3DPathPlanner:
    """
    3D occupancy-grid-aware path planner using OMPL.
    """

    MAX_TRANS_STEP: float = 0.30   # meters per interp step
    MAX_ROT_STEP: float = 0.30     # radians per interp step

    def __init__(self, params: Optional[Dict] = None) -> None:
        params = params or {}

        self.sp = ob.RealVectorStateSpace(3)
        self.set_bounds(params.get("bounds", [-5, 5, -5, 5, -5, 5]))

        self._free_kdt: Optional[KDTree] = None
        self._occ_kdt: Optional[KDTree] = None

        self._use_freegrid: bool = bool(params.get("use_free_grid", False))
        self._use_occgrid: bool = bool(params.get("use_occ_grid", False))
        self._min_dist2occ: float = float(params.get("min_dist2occ", 0.1))
        self._max_dist2free: float = float(params.get("max_dist2free", 0.1))

        # Start/goal
        self.start_pos: Optional[np.ndarray] = None
        self.goal_pos: Optional[np.ndarray] = None
        self.start_quat: Optional[np.ndarray] = None
        self.goal_quat: Optional[np.ndarray] = None

        # OMPL setup
        self.ss = og.SimpleSetup(self.sp)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.sp.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)
        self.solution: List[Dict[str, np.ndarray]] = []

    @property
    def use_freegrid(self) -> bool:
        return self._use_freegrid

    @use_freegrid.setter
    def use_freegrid(self, val: bool) -> None:
        self._use_freegrid = bool(val)

    @property
    def use_occgrid(self) -> bool:
        return self._use_occgrid

    @use_occgrid.setter
    def use_occgrid(self, val: bool) -> None:
        self._use_occgrid = bool(val)

    @property
    def min_dist2occ(self) -> float:
        return self._min_dist2occ

    @min_dist2occ.setter
    def min_dist2occ(self, val: float) -> None:
        self._min_dist2occ = float(val)

    @property
    def max_dist2free(self) -> float:
        return self._max_dist2free

    @max_dist2free.setter
    def max_dist2free(self, val: float) -> None:
        self._max_dist2free = float(val)

    def isfree(self, point: Sequence[float]) -> bool:
        """
        True if sample is within max_dist2free of free voxelgrid (if KDTree set).
        If no free KDTree is set, returns True.
        """
        if self._free_kdt is None:
            return True
        dist, _ = self._free_kdt.query(point)
        return dist < self._max_dist2free

    def isoccupied(self, point: Sequence[float]) -> bool:
        """
        True if sample is within min_dist2occ of occupied voxelgrid (if KDTree set).
        If no occ KDTree is set, returns False.
        """
        if self._occ_kdt is None:
            return False
        dist, _ = self._occ_kdt.query(point)
        return dist < self._min_dist2occ

    def is_state_valid(self, state: ob.State) -> bool:
        p = np.array([state[0], state[1], state[2]], dtype=float)

        in_free = True
        away_from_occ = True

        if self._use_freegrid and self._free_kdt is not None:
            dist, _ = self._free_kdt.query(p)
            in_free = dist < self._max_dist2free

        if self._use_occgrid and self._occ_kdt is not None:
            dist, _ = self._occ_kdt.query(p)
            away_from_occ = dist > self._min_dist2occ

        return in_free and away_from_occ

    def get_bounds(self) -> Dict[str, float]:
        bounds = self.sp.getBounds()
        return {
            "low_x": bounds.low[0],  "high_x": bounds.high[0],
            "low_y": bounds.low[1],  "high_y": bounds.high[1],
            "low_z": bounds.low[2],  "high_z": bounds.high[2],
        }

    def set_bounds(self, bounds_input: Sequence[float]) -> None:
        """
        bounds_input: [low_x, high_x, low_y, high_y, low_z, high_z]
        """
        assert len(bounds_input) == 6, "Bounds must be [low_x, high_x, low_y, high_y, low_z, high_z]"
        lx, hx, ly, hy, lz, hz = map(float, bounds_input)
        assert lx < hx and ly < hy and lz < hz, "Bounds must have low < high for each axis"

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, lx); bounds.setHigh(0, hx)
        bounds.setLow(1, ly); bounds.setHigh(1, hy)
        bounds.setLow(2, lz); bounds.setHigh(2, hz)
        self.sp.setBounds(bounds)

    def update_space(self, free_vx: Optional[np.ndarray] = None, occ_vx: Optional[np.ndarray] = None) -> None:
        if free_vx is not None:
            free_vx = np.asarray(free_vx, dtype=float)
            assert free_vx.ndim == 2 and free_vx.shape[1] == 3, "free_vx should be (N,3)"
            self._free_kdt = KDTree(free_vx)

        if occ_vx is not None:
            occ_vx = np.asarray(occ_vx, dtype=float)
            assert occ_vx.ndim == 2 and occ_vx.shape[1] == 3, "occ_vx should be (M,3)"
            self._occ_kdt = KDTree(occ_vx)

    def update_start_goal(self, start: Union[Dict, np.ndarray], goal: Union[Dict, np.ndarray]) -> bool:
        """
        Accepts either dicts {"pos":..., "quat":...} or 4x4 poses.
        Attempts to nudge invalid start into nearest valid free location if needed.
        """
        # Normalize inputs
        if isinstance(start, np.ndarray) and start.shape == (4, 4):
            start = pose2posquat(start)
        if isinstance(goal, np.ndarray) and goal.shape == (4, 4):
            goal = pose2posquat(goal)

        self.start_pos = np.asarray(start["pos"], dtype=float).copy()
        self.goal_pos  = np.asarray(goal["pos"],  dtype=float).copy()
        self.start_quat = np.asarray(start["quat"], dtype=float).copy()
        self.goal_quat  = np.asarray(goal["quat"],  dtype=float).copy()

        # Nudge start if invalid/occupied
        if (not self.isfree(self.start_pos)) or self.isoccupied(self.start_pos):
            print("Start is not free or too close to occupancy; searching nearest free candidate...")
            if self._free_kdt is None:
                print("No free KDTree available; cannot adjust start.")
                return False

            # k nearest (bounded by dataset size)
            k = min(500, self._free_kdt.n)
            dists, idxs = self._free_kdt.query(self.start_pos, k=k)

            # Ensure arrays for uniform handling
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            found = False
            for d, idx in zip(dists, idxs):
                if d < 3.0 * self._max_dist2free:
                    candidate = self._free_kdt.data[idx]
                    if not self.isoccupied(candidate):
                        print(f"Adjusted start to nearest free at {candidate}")
                        self.start_pos = candidate.astype(float, copy=True)
                        found = True
                        break
            if not found:
                print("No acceptable free start position found within search radius.")
                return False

        start_state = ob.State(self.sp)
        start_state()[0], start_state()[1], start_state()[2] = map(float, self.start_pos)

        goal_state = ob.State(self.sp)
        goal_state()[0], goal_state()[1], goal_state()[2] = map(float, self.goal_pos)

        # The last parameter is the goal region threshold
        self.ss.setStartAndGoalStates(start_state, goal_state, 0.1)
        return True

    def get_start_goal(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            "start": {"pos": self.start_pos, "quat": self.start_quat},
            "goal":  {"pos": self.goal_pos,  "quat": self.goal_quat},
        }

    def get_motion_check_resolution(self) -> float:
        return float(self.ss.getSpaceInformation().getStateValidityCheckingResolution())

    def solve(self, time_limit: float = 5.0, method: str = "rrtstar") -> bool:
        assert time_limit > 0.0, "time_limit must be positive"
        si = self.ss.getSpaceInformation()

        if method == "rrtstar":
            planner = og.RRTstar(si)
        elif method == "rrtconnect":
            planner = og.RRTConnect(si)
        elif method == "rrt":
            planner = og.RRT(si)
        else:
            raise ValueError(f"Unknown planner '{method}'")

        self.ss.setPlanner(planner)
        solved = self.ss.solve(time_limit)
        if solved:
            self.ss.simplifySolution()
            return True
        print("Planner failed to find a solution.")
        return False

    def interpolate_path(
        self,
        num_interp_points: int = 1,
        external_waypoints: Optional[List[Sequence[float]]] = None,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """
        Build a dense, orientation-aware path. If `external_waypoints` is given, uses those.
        Otherwise uses the OMPL solution path. Orientation is:
          - start quat at first,
          - goal quat at last,
          - for intermediates: face the next waypoint.
        """
        try:
            if external_waypoints is None:
                path = self.ss.getSolutionPath()
                if path is None or path.getStateCount() == 0:
                    print("No OMPL path available.")
                    return None
                waypoints = [[s[0], s[1], s[2]] for s in path.getStates()]
            else:
                waypoints = [np.asarray(w, dtype=float).tolist() for w in external_waypoints]
                if len(waypoints) < 2:
                    print("Need at least two waypoints for interpolation.")
                    return None

            # Ensure endpoints match start/goal positions
            waypoints[0]  = np.asarray(self.start_pos, dtype=float).tolist()
            waypoints[-1] = np.asarray(self.goal_pos,  dtype=float).tolist()

            # Build coarse (pos, quat) sequence with
            coarse: List[Dict[str, np.ndarray]] = []
            coarse.append({"pos": np.asarray(self.start_pos, dtype=float),
                           "quat": np.asarray(self.start_quat, dtype=float)})

            for i in range(1, len(waypoints) - 1):
                pos = np.asarray(waypoints[i], dtype=float)
                prev_quat = coarse[-1]["quat"]

                if i == len(waypoints) - 2:
                    # second last: reuse previous orientation 
                    coarse.append({"pos": pos, "quat": prev_quat})
                    continue

                next_pos = np.asarray(waypoints[i + 1], dtype=float)
                v = next_pos - pos
                n = np.linalg.norm(v)
                if n > 1e-8:
                    v /= n
                    # Compute alignment transform for a single origin
                    t_mat = compute_alignment_transforms(
                        origins=[pos],
                        align_vec=v,
                        align_axis=[0, 0, 1],
                        appr_vec=[0, 0, -1],   # CV camera convention
                        appr_axis=[0, 1, 0],
                    )[0]
                    quat = pose2posquat(t_mat)["quat"]
                    coarse.append({"pos": pos, "quat": quat})
                else:
                    coarse.append({"pos": pos, "quat": prev_quat})

            # Append goal
            coarse.append({"pos": np.asarray(self.goal_pos, dtype=float),
                           "quat": np.asarray(self.goal_quat, dtype=float)})

        except Exception as e:
            print(f"No solution found: {e}")
            return None

        # Densify via segment-wise interpolation
        dense: List[Dict[str, np.ndarray]] = []
        for i in range(len(coarse) - 1):
            a, b = coarse[i], coarse[i + 1]

            # Spacing policy based on translation and rotation magnitudes
            tdiff, rdiff = pose_difference(
                posquat2pose(a).reshape(1, 4, 4),
                posquat2pose(b).reshape(1, 4, 4),
            )
            # Extract scalar (pose_difference returns (N,M))
            td = float(np.asarray(tdiff)[0, 0])
            rd = float(np.asarray(rdiff)[0, 0])

            n_trans = int(np.ceil(td / self.MAX_TRANS_STEP))
            n_rot   = int(np.ceil(rd / self.MAX_ROT_STEP))
            n = max(1, n_trans, n_rot, int(num_interp_points))

            seg = interpolate_waypoints(a, b, num_interp_points=n)
            if i == 0:
                dense.extend(seg)
            else:
                # Avoid duplicating previous endpoint
                dense.extend(seg[1:])

        self.solution = dense
        return self.solution

    def get_solution_path(self, return_type: str = "dict"):
        """
        Return the dense path.
        return_type: "dict" -> [{'pos':..., 'quat':...}, ...]
                     "mat"  -> [4x4, 4x4, ...]
        """
        assert return_type in {"dict", "mat"}, "return_type must be 'dict' or 'mat'"
        if return_type == "dict":
            return self.solution
        mats = [posquat2pose(sol) for sol in self.solution]
        return mats
