"""cv_env.py - OpenCV settings that only work if applied before the first `import cv2`.

Import this ahead of cv2 in any entry point: `import cv_env  # noqa: F401`.

With MSMF hardware transforms on, a Windows webcam took ~35 s for the open and again for each
width / height / fps set, ~2.5 minutes in all. Off, the same steps take under half a second, at
the same 720p / 30 fps. Set the variable yourself to override this.
"""

import os

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
