"""geometry.py - pixel column -> horizontal angle (Task 6).

Contract
--------
pixel_to_azimuth(x, image_width, hfov_deg) -> degrees, 0 at the image centre, positive to the
right of the image. The mic-array DOA must use the same zero and sign so angles line up in
av_event_log.SpeakingFusion; agree on this with the audio side before changing it.
"""

from __future__ import annotations


def pixel_to_azimuth(x: float, image_width: int, hfov_deg: float) -> float:
    raise NotImplementedError("Task 6: geometry.pixel_to_azimuth")
