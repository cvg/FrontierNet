import numpy as np
import pywavemap as wave
from pywavemap import InterpolationMode


class WaveMapper:
    def __init__(self, params=None):

        try:
            self.map = wave.Map.create(
                {
                    "type": "hashed_chunked_wavelet_octree",
                    "min_cell_width": {"meters": params["min_cell_width"]},
                }
            )
            self.pipeline = wave.Pipeline(self.map)
            self.pipeline.add_operation(
                {"type": "threshold_map", "once_every": {"seconds": 5.0}}
            )
            self.pipeline.add_integrator(
                "my_integrator",
                {
                    "projection_model": {
                        "type": "pinhole_camera_projector",
                        "width": params["width"],
                        "height": params["height"],
                        "fx": params["fx"],
                        "fy": params["fy"],
                        "cx": params["cx"],
                        "cy": params["cy"],
                    },
                    "measurement_model": {
                        "type": "continuous_ray",
                        "range_sigma": {"meters": 0.05},
                        "scaling_free": 0.2,
                        "scaling_occupied": 0.4,
                    },
                    "integration_method": {
                        "type": "hashed_chunked_wavelet_integrator",
                        "min_range": {"meters": params["min_range"]},
                        "max_range": {"meters": params["max_range"]},
                    },
                },
            )
            self.params = params

            # print all params
            print("WaveMapper initialized with parameters:")
            for key, value in params.items():
                print(f"{key}: {value}")
            self.depth_buffer = []
            self.res = params["resolution"]
            self.query_space = (
                np.mgrid[-20 : 20 : self.res, -20 : 20 : self.res, -5 : 5 : self.res]
                .reshape(3, -1)
                .T
            )
            self.occ_pts = None
            self.free_pts = None

        except Exception as e:
            raise RuntimeError(f"Failed to initialize WaveMapper: {e}")

    def get_parameters(self):
        """
        Get the parameters of the mapper.
        Returns a dictionary of parameters.
        """
        return self.params

    def insert_depth_to_buffer(self, depth, transform):
        """
        Insert depth data into the map using the provided transformation.
        transform: 4x4 numpy, camera IN world frame
        """
        pose = wave.Pose(transform)
        image = wave.Image(np.array(depth).transpose())
        self.depth_buffer.append({"pose": pose, "image": image})

    def integrate_from_buffer(self):
        """
        Integrate all depth data from the buffer into the map.
        """
        for entry in self.depth_buffer:
            self.pipeline.run_pipeline(
                ["my_integrator"], wave.PosedImage(entry["pose"], entry["image"])
            )
        self.depth_buffer.clear()

        self.map.prune()  # Remove map nodes that are no longer needed

    def interpolate_occupancy_grid(self):
        """
        Get the occupancy grid from the map.
        Returns a numpy array of log odds values.
        """
        points_log_odds = self.map.interpolate(
            self.query_space, InterpolationMode.NEAREST
        )
        points_log_odds = points_log_odds.reshape(-1)
        self.occ_pts = self.query_space[points_log_odds > 0.6]
        self.free_pts = self.query_space[points_log_odds < -1e-5]

    def get_occupancy_grid(self):
        return {"occupied": self.occ_pts, "free": self.free_pts}
