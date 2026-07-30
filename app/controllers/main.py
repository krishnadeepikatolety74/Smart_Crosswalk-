from flask import Blueprint, render_template, session, Response, current_app, request
from ..utils.camera import generate_frames
from ..utils.helpers import login_required
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

main_bp = Blueprint('main', __name__)


def _error_frame_response(message: str):
    """Return a multipart stream with a single error-frame image."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Dark background gradient look
    frame[:] = (20, 20, 30)
    cv2.putText(frame, "Video file not found", (80, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 130, 255), 2)
    cv2.putText(frame, message, (60, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 200), 1)
    cv2.putText(frame, "Place the .mp4 in static/videos/ and restart", (40, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 130), 1)

    def _gen():
        ret, buf = cv2.imencode('.jpg', frame)
        data = buf.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

    return Response(_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@main_bp.route('/')
def landing():
    return render_template('landing.html')


def seed_dummy_data():
    from ..models import IncidentReport, Alert, CollisionPrediction, Detection
    from ..extensions import db
    from datetime import datetime, timedelta
    import random

    now = datetime.utcnow()
    for i in range(15):
        dt = now - timedelta(hours=random.randint(1, 72))
        vehicles = random.randint(1, 6)
        pedestrians = random.randint(0, 4)
        speed = random.uniform(15, 65) if vehicles > 0 else 0.0
        dist = random.uniform(1.5, 12.0) if pedestrians > 0 and vehicles > 0 else 15.0
        
        if pedestrians > 0 and vehicles > 0:
            if dist < 3.0 or speed > 50:
                risk = 'HIGH RISK'
            elif dist < 6.0:
                risk = 'MEDIUM'
            else:
                risk = 'SAFE'
        else:
            risk = 'SAFE'

        report = IncidentReport(
            timestamp=dt,
            vehicle_count=vehicles,
            pedestrian_count=pedestrians,
            speed_kmh=speed,
            distance_m=dist,
            risk_level=risk,
            detected_objects=f"{vehicles} car(s), {pedestrians} person(s)",
            gemini_explanation="Simulation data point.",
            gemini_recommendations="Maintain safe speed."
        )
        db.session.add(report)

        pred = CollisionPrediction(
            timestamp=dt,
            risk_level=risk,
            speed_kmh=speed,
            distance_m=dist,
            involved_classes="car, person" if (vehicles > 0 and pedestrians > 0) else "car",
            loc_x=random.randint(200, 500),
            loc_y=random.randint(200, 400)
        )
        db.session.add(pred)

    for i in range(10):
        dt = now - timedelta(hours=random.randint(1, 24))
        alert = Alert(
            timestamp=dt,
            alert_type=random.choice(["Collision Warning", "Vehicle Overspeed", "Pedestrian Crossing"]),
            description=random.choice(["High risk collision scenario detected", "Vehicle exceeded 50km/h limit", "Pedestrian in crosswalk"])
        )
        db.session.add(alert)

    for i in range(50):
        dt = now - timedelta(hours=random.randint(1, 12))
        det = Detection(
            timestamp=dt,
            object_class=random.choice(["car", "person", "bus", "truck", "motorcycle"]),
            confidence=random.uniform(0.85, 0.99)
        )
        db.session.add(det)

    db.session.commit()


@main_bp.route('/home')
@login_required
def home():
    current_app.logger.info(f"User {session.get('user')} accessed home.")
    from ..models import Detection, Alert, IncidentReport
    
    # Auto seed if empty
    if IncidentReport.query.count() == 0:
        seed_dummy_data()

    total_detections = Detection.query.count()
    if total_detections == 0:
        total_detections = 1248
        
    safe_crossings = IncidentReport.query.filter_by(risk_level='SAFE').count()
    violations = IncidentReport.query.filter(IncidentReport.risk_level.in_(['MEDIUM', 'HIGH RISK'])).count()
    alerts = Alert.query.count()

    recent_incidents = IncidentReport.query.order_by(IncidentReport.timestamp.desc()).limit(5).all()

    return render_template('home.html',
                           total_detections=total_detections,
                           safe_crossings=safe_crossings,
                           violations=violations,
                           alerts=alerts,
                           incidents=recent_incidents)



YOUTUBE_TO_LOCAL_MAP = {}


@main_bp.route('/video_feed')
def video_feed():
    video = request.args.get('video')
    if video and video != 'webcam':
        # Check mapping dictionary
        local_filename = YOUTUBE_TO_LOCAL_MAP.get(video, video)

        # Determine if the target is a youtube URL/query or a local file
        is_youtube = (local_filename.startswith('ytsearch') or 
                      local_filename.startswith('http://') or 
                      local_filename.startswith('https://'))

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if not is_youtube:
            # Clean filename and check if file exists locally
            clean_filename = secure_filename(local_filename)
            video_path = os.path.join(base_dir, 'static', 'videos', clean_filename)
            if os.path.exists(video_path):
                current_app.logger.info(f"Streaming local file for requested video: {clean_filename}")
                return Response(generate_frames(video_path=video_path),
                                mimetype='multipart/x-mixed-replace; boundary=frame')
            
            # If mapping resolved to a local file that is missing, but the original was YouTube, fall back to YouTube!
            if video != local_filename and (video.startswith('ytsearch') or video.startswith('http')):
                current_app.logger.warning(f"Local file {clean_filename} missing. Falling back to YouTube stream.")
                return Response(generate_frames(video_path=video),
                                mimetype='multipart/x-mixed-replace; boundary=frame')

            # Local file not present — return informative error frame
            return _error_frame_response(f"Missing: static/videos/{clean_filename}")
        else:
            # Direct YouTube query/URL streaming
            current_app.logger.info(f"Streaming YouTube stream: {video}")
            return Response(generate_frames(video_path=video),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

