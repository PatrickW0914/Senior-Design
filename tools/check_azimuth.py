#!/usr/bin/env python3
"""check_azimuth.py - Task 6 floor test: does the computed azimuth match tape marks at known angles?

  python tools/check_azimuth.py --source 0 --marks -30 0 30

Put tape on the floor at the given angles from the camera (measure: at distance d straight out,
a mark at angle a is d * tan(a) to the side; left of the camera is negative). Stand on each mark
and read the azimuth printed on your face box; it should match the mark within a degree or two.
Vertical lines show where each mark angle lands in the image. Also works on a clip.
Uses camera.calibration from config.yaml when set, otherwise camera.hfov_deg.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv_env  # noqa: F401, E402
import cv2  # noqa: E402

from av_event_log import SessionClock  # noqa: E402
from capture import VideoSource  # noqa: E402
from config import load_config  # noqa: E402
from detect import FaceDetector  # noqa: E402
from geometry import AzimuthModel  # noqa: E402
from main import parse_source  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="0", help="camera index or clip (default 0)")
    p.add_argument("--marks", type=float, nargs="*", default=[-30.0, 0.0, 30.0], help="mark angles in degrees")
    args = p.parse_args(argv)
    cfg = load_config()
    detector = FaceDetector(cfg["detect"])
    f = cv2.FONT_HERSHEY_SIMPLEX
    with VideoSource(parse_source(args.source), cfg["camera"], SessionClock()) as video:
        model = AzimuthModel.from_config(cfg["camera"], video.width, video.height)
        print(f"{video.width}x{video.height}  model: {model.source}  hfov {model.hfov_deg:.1f} deg  "
              f"fx {model.fx:.1f} px  cx {model.cx:.1f}")
        last_print = -1.0
        for t, frame in video:
            faces = detector.detect(frame)
            for a in args.marks:
                x = int(round(model.x_at_azimuth(a)))
                cv2.line(frame, (x, 0), (x, frame.shape[0]), (0, 255, 255), 2)
                cv2.putText(frame, f"{a:+.0f} deg", (x + 6, 40), f, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            for d in faces:
                x1, y1, x2, y2 = (int(v) for v in d.bbox)
                az = model.azimuth(d.cx, 0.5 * (y1 + y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"{az:+.1f} deg", (x1, max(30, y1 - 10)), f, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
                if t - last_print >= 0.5:
                    print(f"t={t:7.2f}  face x={d.cx:7.1f}  azimuth={az:+6.1f} deg")
                    last_print = t
            cv2.imshow("check_azimuth (q quits)", cv2.resize(frame, None, fx=min(1.0, 1280 / frame.shape[1]),
                                                              fy=min(1.0, 1280 / frame.shape[1])))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
