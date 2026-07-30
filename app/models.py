from .extensions import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class BaseModel(db.Model):
    __abstract__ = True
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class User(BaseModel):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), default='traffic_authority')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Detection(BaseModel):
    __tablename__ = 'detections'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    object_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255))

class Analytics(BaseModel):
    __tablename__ = 'analytics'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True)
    total_detections = db.Column(db.Integer, default=0)
    risk_level = db.Column(db.String(20), default='Low')


class Setting(BaseModel):
    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.String(255), nullable=False)

class CollisionPrediction(BaseModel):
    __tablename__ = 'collision_predictions'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    risk_level = db.Column(db.String(20), nullable=False)
    speed_kmh = db.Column(db.Float, nullable=False)
    distance_m = db.Column(db.Float, nullable=False)
    involved_classes = db.Column(db.String(100))
    loc_x = db.Column(db.Integer, default=0)
    loc_y = db.Column(db.Integer, default=0)

class Alert(BaseModel):
    __tablename__ = 'alerts'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    alert_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(255))

class IncidentReport(BaseModel):
    __tablename__ = 'incident_reports'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    vehicle_count = db.Column(db.Integer, nullable=False)
    pedestrian_count = db.Column(db.Integer, nullable=False)
    speed_kmh = db.Column(db.Float, nullable=False)
    distance_m = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    detected_objects = db.Column(db.String(255))
    gemini_explanation = db.Column(db.Text)
    gemini_recommendations = db.Column(db.Text)
    pdf_path = db.Column(db.String(255))

class TrackingHistory(BaseModel):
    __tablename__ = 'tracking_history'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    object_id = db.Column(db.Integer, nullable=False)
    object_class = db.Column(db.String(50), nullable=False)
    speed_kmh = db.Column(db.Float, default=0.0)
    direction = db.Column(db.String(50), default='')
    distance_m = db.Column(db.Float, default=0.0)

class TrafficSignalHistory(BaseModel):
    __tablename__ = 'traffic_signal_history'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    state = db.Column(db.String(20), nullable=False)
    object_id = db.Column(db.Integer, nullable=False)


class TrafficSnapshot(BaseModel):
    __tablename__ = 'traffic_snapshots'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    location = db.Column(db.String(100), nullable=False)
    congestion_level = db.Column(db.String(50), nullable=False)
    travel_time_mins = db.Column(db.Integer, default=0)
    incident_count = db.Column(db.Integer, default=0)
