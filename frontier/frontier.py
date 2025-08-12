import numpy as np
from typing import Optional, Dict, Any


class Frontier:
    def __init__(self, frontier_feature: Optional[np.ndarray] = None):
        """
        Initialize a Frontier object.
        """
        # initialize all known feature slots
        self.features: Dict[str, Any] = {
            "3d_pos": None,  # 3D position of the frontier
            "gain": None,  # initial information gain (volume in m^3 from network prediction)
            "direct_angle": None,  # 2D viewing direction on the image of the frontier
            "pixel_pos": None,  # 2D pixel position of the frontier
            "vd": None,  # 3D view direction in world frame of the frontier
            "valid_flag": True,  # whether the frontier is valid (default is True)
            "u_gain": None,  # the updated gain of the frontier (initially the same as gain)
            "utility": None,  # utility score of the frontier (only valid in the manager)
            "id": 0,  # unique identifier for the frontier (only valid in the manager)
            "parent_ids": [],  # list of parent frontier IDs (only valid with the graph in the manager)
            "6d_pose": None,  # placeholder for future 6D pose
        }

        if frontier_feature is not None:
            arr = np.asarray(frontier_feature, dtype=float)
            if arr.shape != (10,):
                raise ValueError("Frontier feature must be a 10-element vector.")

            pos3d, gain, direct_angle, px, py, vx, vy, vz = (
                arr[:3],
                float(arr[3]),
                float(arr[4]),
                float(arr[5]),
                float(arr[6]),
                float(arr[7]),
                float(arr[8]),
                float(arr[9]),
            )

            self.features["3d_pos"] = pos3d
            self.features["gain"] = gain
            self.features["direct_angle"] = direct_angle
            self.features["pixel_pos"] = np.array([px, py])
            self.features["vd"] = np.array([vx, vy, vz])
            self.features["u_gain"] = gain
            # valid_flag stays True by default
            # id, utility, parent_ids remain at their defaults
            # 6d_pose remains None

    @property
    def pose6d(self):
        """
        Placeholder for 6D pose, not used in this class.
        """
        return self.features.get("6d_pose", None)

    @pose6d.setter
    def pose6d(self, val):
        if val.shape != (4, 4) and val is not None:
            raise ValueError("6D pose must be a 4x4 matrix.")
        self.features["6d_pose"] = val

    @property
    def id(self) -> int:
        return self.features["id"]

    @id.setter
    def id(self, val: int):
        self.features["id"] = val

    @property
    def pos3d(self) -> Optional[np.ndarray]:
        return self.features["3d_pos"]

    @pos3d.setter
    def pos3d(self, val: np.ndarray):
        self.features["3d_pos"] = val

    @property
    def gain(self) -> Optional[float]:
        return self.features["gain"]

    @gain.setter
    def gain(self, val: float):
        self.features["gain"] = val

    @property
    def direct_angle(self) -> Optional[float]:
        return self.features["direct_angle"]

    @direct_angle.setter
    def direct_angle(self, val: float):
        self.features["direct_angle"] = val

    @property
    def pixel_pos(self) -> Optional[np.ndarray]:
        return self.features["pixel_pos"]

    @pixel_pos.setter
    def pixel_pos(self, val: np.ndarray):
        self.features["pixel_pos"] = val

    @property
    def view_direction(self) -> Optional[np.ndarray]:
        return self.features["vd"]

    @view_direction.setter
    def view_direction(self, val: np.ndarray):
        self.features["vd"] = val

    @property
    def u_gain(self) -> Optional[float]:
        return self.features["u_gain"]

    @u_gain.setter
    def u_gain(self, val: float):
        self.features["u_gain"] = val

    @property
    def utility(self) -> Optional[float]:
        return self.features["utility"]

    @utility.setter
    def utility(self, val: float):
        self.features["utility"] = val

    @property
    def is_valid(self) -> bool:
        return self.features["valid_flag"]

    def set_invalid(self):
        self.features["valid_flag"] = False

    def set_valid(self):
        self.features["valid_flag"] = True

    @property
    def parent_ids(self) -> list:
        return self.features["parent_ids"]

    @parent_ids.setter
    def parent_ids(self, val: list):
        if not isinstance(val, list):
            raise TypeError("parent_ids must be a list.")

        seen = set()
        for i in val:
            if not isinstance(i, int):
                raise TypeError(
                    f"parent_ids must be a list of integers; got {type(i).__name__!r}."
                )
            if i in seen:
                raise ValueError(f"parent_ids must be unique; duplicate {i!r} found.")
            seen.add(i)

        # All checks passed:
        self.features["parent_ids"] = val

    def get_feature_dict(self) -> Dict[str, Any]:
        return self.features

    def is_in_bbox(self, bbox: np.ndarray) -> bool:
        """
        Check if 3D position is inside a given bounding box.
        bbox: [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        if self.pos3d is None:
            return False
        return (
            bbox[0] <= self.pos3d[0] <= bbox[1]
            and bbox[2] <= self.pos3d[1] <= bbox[3]
            and bbox[4] <= self.pos3d[2] <= bbox[5]
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (arrays become lists, Nones kept as None)."""
        def _list(x): return None if x is None else np.asarray(x, dtype=float).tolist()
        return {
            "id": self.id,
            "3d_pos": _list(self.pos3d),
            "gain": self.gain,
            "direct_angle": self.direct_angle,
            "pixel_pos": _list(self.pixel_pos),
            "vd": _list(self.view_direction),
            "valid_flag": self.is_valid,
            "u_gain": self.u_gain,
            "utility": self.utility,
            "parent_ids": list(self.parent_ids),
        }

    def from_dict(self, data: Dict[str, Any]):
        self.id = data.get("id", 0)
        self.pos3d = data.get("3d_pos", None)
        self.gain = data.get("gain", None)
        self.direct_angle = data.get("direct_angle", None)
        self.pixel_pos = data.get("pixel_pos", None)
        self.view_direction = data.get("vd", None)
        self.set_valid() if data.get("valid_flag", True) else self.set_invalid()
        self.u_gain = data.get("u_gain", self.gain)
        self.utility = data.get("utility", None)
        self.parent_ids = data.get("parent_ids", [])

    def __repr__(self):
        return (f"Frontier(id={self.id}, pos3d={self.pos3d}, gain={self.gain}, "
                f"direct_angle={self.direct_angle}, pixel_pos={self.pixel_pos}, "
                f"vd={self.view_direction}, valid_flag={self.is_valid}, "
                f"u_gain={self.u_gain}, utility={self.utility}, parent_ids={self.parent_ids})")