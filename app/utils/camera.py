import cv2
import logging
import time
import os
import math
import numpy as np
from .analytics_engine import AnalyticsEngine, global_crosswalk_state
from .model_loader import get_yolo_model

# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 9]


def _safe_db_write(app_obj, fn):
    """Safely execute a DB write inside an app context, silently ignoring errors."""
    if not app_obj:
        return
    try:
        with app_obj.app_context():
            fn()
    except Exception as e:
        logging.debug(f"DB write skipped in generator: {e}")


def generate_frames(video_path=None):
    """Generator that yields MJPEG frames for live camera feed."""
    prev_time = 0
    engine = AnalyticsEngine()

    # Capture the app object safely — this runs in the request context
    try:
        from flask import current_app
        app_obj = current_app._get_current_object()
    except RuntimeError:
        app_obj = None

    yolo_model = get_yolo_model()

    # ── Source selection ────────────────────────────────────────────────
    source = video_path if video_path else 0
    
    if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://') or source.startswith('ytsearch')):
        try:
            import yt_dlp
            logging.info(f"Resolving YouTube URL/Query: {source}")
            ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True, 'noplaylist': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                if 'entries' in info:
                    source = info['entries'][0].get('url', source)
                else:
                    source = info.get('url', source)
        except Exception as e:
            logging.error(f"Failed to resolve YouTube URL: {e}")

    camera = cv2.VideoCapture(source)

    # If webcam fails, look for any .mp4 in static/videos/ as a fallback
    if not camera.isOpened() and source == 0:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        videos_dir = os.path.join(base_dir, 'static', 'videos')
        fallback_videos = []
        if os.path.exists(videos_dir):
            fallback_videos = sorted(
                [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) if f.lower().endswith('.mp4')]
            )
        if fallback_videos:
            source = fallback_videos[0]
            camera = cv2.VideoCapture(source)
            video_path = source  # enable looping
            logging.info(f"Webcam unavailable, using fallback video: {source}")

    # ── Synthetic simulator when no real source is available ─────────────
    if not camera.isOpened():
        logging.warning("No camera or fallback video available. Starting synthetic traffic simulation feed.")
        frame_idx = 0
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Dark grid background
            for x in range(0, 640, 40):
                cv2.line(frame, (x, 0), (x, 480), (25, 25, 28), 1)
            for y in range(0, 480, 40):
                cv2.line(frame, (0, y), (640, y), (25, 25, 28), 1)

            # Crosswalk
            cv2.rectangle(frame, (180, 160), (460, 320), (45, 45, 45), -1)
            for cy_stripe in range(170, 320, 20):
                cv2.rectangle(frame, (200, cy_stripe), (440, cy_stripe + 10), (140, 140, 140), -1)

            # Animated pedestrian
            ped_x = int(200 + (frame_idx % 220))
            ped_y = 230

            # Animated car
            car_x = 320
            car_y = int(50 + (frame_idx * 3.5) % 360)

            dist_px = math.sqrt((ped_x - car_x) ** 2 + (ped_y - car_y) ** 2)
            dist_m = dist_px * 0.05

            # Draw objects - thicker lines and larger fonts for clear visibility
            cv2.rectangle(frame, (ped_x, ped_y), (ped_x + 25, ped_y + 50), (166, 227, 161), 3)
            cv2.putText(frame, "Person 95%", (ped_x, ped_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (166, 227, 161), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (car_x - 25, car_y - 35), (car_x + 25, car_y + 35), (137, 180, 250), 3)
            cv2.putText(frame, "Car 89%", (car_x - 25, car_y - 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (137, 180, 250), 2, cv2.LINE_AA)
            cv2.line(frame, (ped_x + 10, ped_y + 22), (car_x, car_y), (255, 255, 255), 1)
            cv2.putText(frame, f"{dist_m:.1f}m", ((ped_x + 10 + car_x) // 2, (ped_y + 22 + car_y) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Risk level
            risk = "SAFE"
            risk_color = (166, 227, 161)
            if dist_m < 3.0:
                risk = "HIGH RISK"
                risk_color = (243, 139, 168)
                cv2.putText(frame, "COLLISION ALERT", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_color, 2)
            elif dist_m < 5.0:
                risk = "MEDIUM"
                risk_color = (250, 179, 135)

            # Update global state
            in_cw = 200 < ped_x < 420
            global_crosswalk_state["status"] = "Crosswalk Occupied" if in_cw else "Crosswalk Empty"
            global_crosswalk_state["pedestrian_count"] = 1 if in_cw else 0

            # Overlays
            curr_time = time.time()
            fps_val = 1 / (curr_time - prev_time) if prev_time else 25
            prev_time = curr_time
            cv2.putText(frame, f"Crosswalk: {global_crosswalk_state['status']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Risk: {risk}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
            cv2.putText(frame, f"FPS: {int(fps_val)}", (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (166, 227, 161), 1)
            cv2.putText(frame, "[ AI Simulator — No Camera Detected ]", (100, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

            # Periodic DB write (every 100 frames)
            if app_obj and frame_idx % 100 == 0:
                def _write_sim_alert(r=risk):
                    from ..models import Alert
                    from ..extensions import db
                    alert = Alert(alert_type="Simulated Event", description=f"Synthetic activity: {r}")
                    db.session.add(alert)
                    db.session.commit()
                _safe_db_write(app_obj, _write_sim_alert)

            frame_idx += 1
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.04)  # ~25 FPS cap
        return  # generator ends here for synthetic path

    # ── Real Camera / Video loop ──────────────────────────────────────────
    logging.info(f"Camera/video opened: {source}")
    consecutive_failures = 0
    max_failures = 10

    while True:
        success, frame = camera.read()
        if not success:
            consecutive_failures += 1
            if video_path:  # loop the video
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                consecutive_failures = 0
                continue
            if consecutive_failures >= max_failures:
                logging.warning("Camera read failed repeatedly, stopping generator.")
                break
            time.sleep(0.05)
            continue

        consecutive_failures = 0

        if yolo_model:
            try:
                results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                # Custom bounding boxes and overlays are drawn inside engine.process_frame
                frame = engine.process_frame(frame, results, app=app_obj)
            except Exception as e:
                logging.error(f"YOLO inference error: {e}")

        # Sync FPS variables
        curr_time = time.time()
        fps_val = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        if video_path:
            time.sleep(0.033)  # ~30 FPS for video playback

    camera.release()
    logging.info("Camera released.")
