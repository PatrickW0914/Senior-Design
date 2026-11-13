"""geometry.py - pixel column -> horizontal angle (Task 6).

Contract
--------
pixel_to_azimuth(x, image_width, hfov_deg) -> degrees. 0 is straight ahead (the image centre),
negative is to the camera's left (the left half of the image), positive to its right, and the
image edges are at -hfov/2 and +hfov/2. Pinhole model, azimuth = atan((x - cx) / fx): the
angle per pixel shrinks towards the edges, so this is not a linear scale. It fills
FaceObs.azimuth_deg; the mic array's DOA must use the same zero and sign.

AzimuthModel is the calibrated version: focal length and principal point from
tools/calibrate.py (data/calibration/*.yaml), lens distortion undone, intrinsics scaled to the
frame size in use. AzimuthModel.from_config() uses the calibration file camera.calibration
names, or camera.hfov_deg when there is none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from config import repo_path


def focal_px(image_width: int, hfov_deg: float) -> float:
    """Focal length in pixels of a camera that sees hfov_deg across image_width pixels."""
    return 0.5 * image_width / math.tan(math.radians(hfov_deg) / 2)


def hfov_deg_from_focal(image_width: int, fx: float) -> float:
    return 2 * math.degrees(math.atan(0.5 * image_width / fx))


def pixel_to_azimuth(x: float, image_width: int, hfov_deg: float) -> float:
    """Horizontal angle of pixel column x: 0 at the centre, -hfov/2 at x=0, +hfov/2 at x=width."""
    return math.degrees(math.atan((x - 0.5 * image_width) / focal_px(image_width, hfov_deg)))


@dataclass
class AzimuthModel:
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    dist: np.ndarray | None = None       # OpenCV distortion coefficients for this fx/fy/cx/cy
    source: str = "hfov"

    # --- constructors ---------------------------------------------------------------------
    @classmethod
    def from_hfov(cls, image_width: int, image_height: int, hfov_deg: float) -> AzimuthModel:
        f = focal_px(image_width, hfov_deg)
        return cls(f, f, 0.5 * image_width, 0.5 * image_height, image_width, image_height, None, "hfov")

    @classmethod
    def from_calibration(cls, path: str | Path, image_width: int, image_height: int) -> AzimuthModel:
        """Load tools/calibrate.py output and scale it to the frame size in use.

        Scaling assumes the camera shows the same field of view at both sizes (the usual case for
        a webcam at 720p vs 1080p); a mode that crops the sensor would need its own calibration.
        """
        path = repo_path(path)
        with open(path, encoding="utf-8") as fh:
            cal = yaml.safe_load(fh)
        k = np.asarray(cal["camera_matrix"], dtype=float)
        sx, sy = image_width / cal["image_width"], image_height / cal["image_height"]
        if abs(sx / sy - 1) > 0.02:
            raise ValueError(f"{path.name} was made at {cal['image_width']}x{cal['image_height']}; "
                             f"{image_width}x{image_height} has a different aspect ratio, so it is a "
                             "different crop: calibrate at this size")
        dist = np.asarray(cal.get("dist_coeffs") or [], dtype=float)
        return cls(k[0, 0] * sx, k[1, 1] * sy, k[0, 2] * sx, k[1, 2] * sy, image_width, image_height,
                   dist if dist.size else None, str(path.name))

    @classmethod
    def from_config(cls, camera_cfg: dict, image_width: int, image_height: int) -> AzimuthModel:
        if camera_cfg.get("calibration"):
            return cls.from_calibration(camera_cfg["calibration"], image_width, image_height)
        return cls.from_hfov(image_width, image_height, float(camera_cfg["hfov_deg"]))

    # --- angles ---------------------------------------------------------------------------
    @property
    def hfov_deg(self) -> float:
        return hfov_deg_from_focal(self.image_width, self.fx)

    def normalized(self, x: float, y: float | None = None) -> tuple[float, float]:
        """Pixel -> normalized image coordinates (x/z, y/z), distortion removed when known."""
        if y is None:
            y = self.cy
        if self.dist is not None:
            import cv2
            k = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=float)
            pt = cv2.undistortPoints(np.array([[[float(x), float(y)]]], dtype=float), k, self.dist)
            return float(pt[0, 0, 0]), float(pt[0, 0, 1])
        return (x - self.cx) / self.fx, (y - self.cy) / self.fy

    def azimuth(self, x: float, y: float | None = None) -> float:
        """Degrees left (-) / right (+) of straight ahead for pixel column x (row y helps undistort)."""
        return math.degrees(math.atan(self.normalized(x, y)[0]))

    def elevation(self, x: float, y: float) -> float:
        """Degrees above (+) / below (-) straight ahead. Not used by the pipeline yet."""
        return -math.degrees(math.atan(self.normalized(x, y)[1]))

    def x_at_azimuth(self, azimuth_deg: float) -> float:
        """Pixel column where a given azimuth lands (distortion ignored); for drawing angle marks."""
        return self.cx + self.fx * math.tan(math.radians(azimuth_deg))
