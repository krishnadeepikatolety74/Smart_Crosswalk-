from flask import Blueprint, render_template, redirect, url_for, request, current_app, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import uuid
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from ..utils.image_processor import process_image
from ..utils.video_processor import process_video
from ..extensions import db
from ..models import Detection, Alert, CollisionPrediction, IncidentReport, Setting
from ..utils.gemini_helper import generate_incident_explanation
from ..utils.analytics_engine import global_traffic_state
from ..utils.pdf_generator import generate_pdf_report
from ..utils.helpers import login_required, role_required

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@dashboard_bp.route('/')
@login_required
def dashboard():
    return redirect(url_for('dashboard.live_camera'))

@dashboard_bp.route('/alerts/data')
@login_required
def get_alerts():
    alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(10).all()
    return jsonify([{'type': a.alert_type, 'desc': a.description, 'time': a.timestamp.strftime('%H:%M:%S')} for a in alerts])

@dashboard_bp.route('/users')
@login_required
@role_required('admin')
def users():
    from ..models import User
    current_app.logger.info("Users page accessed")
    users_list = User.query.all()
    return render_template('users.html', users=users_list)

@dashboard_bp.route('/live-camera')
@login_required
@role_required('traffic_authority', 'admin')
def live_camera():
    current_app.logger.info("Live camera dashboard accessed")
    return render_template('live_camera.html')

@dashboard_bp.route('/upload-video', methods=['GET', 'POST'])
@login_required
def upload_video():
    if request.method == 'POST':
        is_demo = request.form.get('demo') == 'true'
        
        if not is_demo and 'video' not in request.files:
            current_app.logger.warning("Upload video failed: no video in request")
            return render_template('upload_video.html', error="No video file found in the request upload.")
            
        file = None
        if not is_demo:
            file = request.files['video']
            if file.filename == '':
                current_app.logger.warning("Upload video failed: empty filename")
                return render_template('upload_video.html', error="Selected file has an empty filename.")
            
        if is_demo or file:
            unique_id = str(uuid.uuid4())[:8] if not is_demo else "demo"
            base_dir = os.path.dirname(current_app.root_path)
            upload_dir = os.path.join(base_dir, 'static', 'uploads')
            processed_dir = os.path.join(base_dir, 'static', 'processed')
            
            os.makedirs(upload_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            if is_demo:
                filename = "demo_hyderabad.mp4"
                upload_path = os.path.join(base_dir, 'static', 'videos', 'hyderabad.mp4')
                if not os.path.exists(upload_path):
                    videos_dir = os.path.join(base_dir, 'static', 'videos')
                    fallback_videos = [f for f in os.listdir(videos_dir) if f.lower().endswith('.mp4')]
                    if fallback_videos:
                        upload_path = os.path.join(videos_dir, fallback_videos[0])
                        filename = fallback_videos[0]
            else:
                filename = secure_filename(file.filename)
                upload_path = os.path.join(upload_dir, f"{unique_id}_{filename}")
                file.save(upload_path)
                
            save_name = f"{unique_id}_{filename}"
            processed_path = os.path.join(processed_dir, save_name)
            processed_url = url_for('static', filename=f'processed/{save_name}')
            
            current_app.logger.info(f"Processing video: {filename}")
            results = process_video(upload_path, processed_path)
            
            if results.get('success'):
                for det in results['detections']:
                    new_detection = Detection(
                        object_class=det['class_name'],
                        confidence=det['confidence'],
                        image_path=processed_url
                    )
                    db.session.add(new_detection)
                db.session.commit()
                current_app.logger.info(f"Video {filename} processed successfully")
                
                return render_template(
                    'upload_video.html',
                    processed_video_url=processed_url,
                    processed_snapshot_url='/static' + results.get('snapshot_url', ''),
                    vehicle_trend=results.get('vehicle_trend', []),
                    risk_trend=results.get('risk_trend', []),
                    total_count=results['count'],
                    detections=results['detections']
                )
            else:
                current_app.logger.error(f"Video processing failed: {results.get('error')}")
                return render_template('upload_video.html', error=results.get('error', 'Error occurred during AI processing.'))
                
    return render_template('upload_video.html')

@dashboard_bp.route('/upload-image', methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            current_app.logger.warning("Upload image failed: no image in request")
            return render_template('upload_image.html', error="No image file found in the request upload.")
            
        file = request.files['image']
        if file.filename == '':
            current_app.logger.warning("Upload image failed: empty filename")
            return render_template('upload_image.html', error="Selected file has an empty filename.")
            
        if file:
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())[:8]
            
            base_dir = os.path.dirname(current_app.root_path)
            upload_dir = os.path.join(base_dir, 'static', 'uploads')
            processed_dir = os.path.join(base_dir, 'static', 'processed')
            
            os.makedirs(upload_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            save_name = f"{unique_id}_{filename}"
            upload_path = os.path.join(upload_dir, save_name)
            processed_path = os.path.join(processed_dir, save_name)
            
            processed_url = url_for('static', filename=f'processed/{save_name}')
            file.save(upload_path)
            current_app.logger.info(f"Processing uploaded image: {filename}")
            
            results = process_image(upload_path, processed_path)
            
            if results.get('success'):
                for det in results['detections']:
                    new_detection = Detection(
                        object_class=det['class_name'],
                        confidence=det['confidence'],
                        image_path=processed_url
                    )
                    db.session.add(new_detection)
                
                # Store results into SQLite as IncidentReport
                incident = IncidentReport(
                    vehicle_count=results.get('vehicle_count', 0),
                    pedestrian_count=results.get('pedestrian_count', 0),
                    speed_kmh=0.0,
                    distance_m=results.get('min_dist_m', 0.0),
                    risk_level=results.get('risk_level', 'SAFE'),
                    detected_objects=", ".join([d['class_name'] for d in results.get('detections', [])]),
                    gemini_explanation=results.get('gemini_explanation', ''),
                    gemini_recommendations=results.get('gemini_recommendation', '')
                )
                db.session.add(incident)
                db.session.commit()
                current_app.logger.info(f"Image {filename} processed successfully")
                
                return render_template(
                    'upload_image.html',
                    processed_image_url=processed_url,
                    total_count=results['count'],
                    detections=results['detections'],
                    risk_level=results['risk_level'],
                    gemini_explanation=results.get('gemini_explanation'),
                    gemini_recommendation=results.get('gemini_recommendation')
                )
            else:
                current_app.logger.error(f"Image processing failed: {results.get('error')}")
                return render_template('upload_image.html', error=results.get('error', 'Error occurred during AI processing.'))
                
    return render_template('upload_image.html')

@dashboard_bp.route('/history')
@login_required
@role_required('traffic_authority', 'admin')
def history():
    current_app.logger.info("History page accessed")
    query = IncidentReport.query
    
    risk_filter = request.args.get('risk')
    if risk_filter:
        query = query.filter_by(risk_level=risk_filter)
        
    date_filter = request.args.get('date')
    if date_filter:
        query = query.filter(db.func.date(IncidentReport.timestamp) == date_filter)
        
    incidents = query.order_by(IncidentReport.timestamp.desc()).all()
    return render_template('history.html', incidents=incidents)

@dashboard_bp.route('/analytics')
@login_required
@role_required('admin')
def analytics():
    current_app.logger.info("Analytics page accessed")
    today = datetime.utcnow().date()
    total_vehicles = Detection.query.filter(
        Detection.object_class.in_(['car', 'bus', 'truck', 'motorcycle', 'bicycle']),
        db.func.date(Detection.timestamp) == today
    ).count()
    total_pedestrians = Detection.query.filter(
        Detection.object_class == 'person',
        db.func.date(Detection.timestamp) == today
    ).count()
    
    high_risk = CollisionPrediction.query.filter_by(risk_level='HIGH RISK').count()
    medium_risk = CollisionPrediction.query.filter_by(risk_level='MEDIUM').count()
    safe_events = CollisionPrediction.query.filter_by(risk_level='SAFE').count()
    
    predictions = CollisionPrediction.query.all()
    avg_speed = sum([p.speed_kmh for p in predictions]) / len(predictions) if predictions else 0
    
    # Bar Chart
    fig_bar, ax_bar = plt.subplots(figsize=(5,3))
    types = ['Vehicles', 'Pedestrians']
    counts = [total_vehicles, total_pedestrians]
    ax_bar.bar(types, counts, color=['#9F7AEA', '#ED64A6'])
    ax_bar.set_title("Traffic Volume")
    plt.tight_layout()
    buf_bar = io.BytesIO()
    fig_bar.savefig(buf_bar, format='png', transparent=True)
    buf_bar.seek(0)
    bar_chart = base64.b64encode(buf_bar.getvalue()).decode('utf-8')
    plt.close(fig_bar)
    
    # Pie Chart
    fig_pie, ax_pie = plt.subplots(figsize=(4,4))
    risks = ['High Risk', 'Medium', 'Safe']
    risk_counts = [high_risk, medium_risk, safe_events]
    has_data = sum(risk_counts) > 0
    ax_pie.pie(risk_counts if has_data else [1], labels=risks if has_data else ["No Data"], colors=['#F56565', '#ECC94B', '#48BB78'] if has_data else ['#e2e8f0'], autopct='%1.1f%%' if has_data else '')
    ax_pie.set_title("Risk Distribution")
    plt.tight_layout()
    buf_pie = io.BytesIO()
    fig_pie.savefig(buf_pie, format='png', transparent=True)
    buf_pie.seek(0)
    pie_chart = base64.b64encode(buf_pie.getvalue()).decode('utf-8')
    plt.close(fig_pie)
    
    # Line Chart
    fig_line, ax_line = plt.subplots(figsize=(5,3))
    if predictions:
        speeds = [p.speed_kmh for p in predictions[-20:]]
        ax_line.plot(speeds, marker='o', color='#805AD5')
        ax_line.set_title("Recent Speed Trend (km/h)")
    else:
        ax_line.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center')
    plt.tight_layout()
    buf_line = io.BytesIO()
    fig_line.savefig(buf_line, format='png', transparent=True)
    buf_line.seek(0)
    line_chart = base64.b64encode(buf_line.getvalue()).decode('utf-8')
    plt.close(fig_line)
    
    # Heat Map
    import seaborn as sns
    import pandas as pd
    fig_heat, ax_heat = plt.subplots(figsize=(5,3))
    collisions = CollisionPrediction.query.filter(CollisionPrediction.loc_x != 0).all()
    if collisions:
        df = pd.DataFrame([{'x': c.loc_x, 'y': c.loc_y} for c in collisions])
        sns.kdeplot(data=df, x='x', y='y', fill=True, cmap='rocket_r', ax=ax_heat)
        ax_heat.set_title("Dangerous Crossing Zones")
        ax_heat.invert_yaxis()
    else:
        ax_heat.text(0.5, 0.5, "No spatial data available", horizontalalignment='center', verticalalignment='center')
        ax_heat.set_title("Dangerous Crossing Zones")
    plt.tight_layout()
    buf_heat = io.BytesIO()
    fig_heat.savefig(buf_heat, format='png', transparent=True)
    buf_heat.seek(0)
    heat_map = base64.b64encode(buf_heat.getvalue()).decode('utf-8')
    plt.close(fig_heat)
    
    return render_template(
        'analytics.html',
        total_vehicles=total_vehicles,
        total_pedestrians=total_pedestrians,
        high_risk=high_risk,
        safe=safe_events,
        avg_speed=int(avg_speed),
        bar_chart=bar_chart,
        pie_chart=pie_chart,
        line_chart=line_chart,
        heat_map=heat_map
    )

@dashboard_bp.route('/reports')
@login_required
@role_required('admin')
def reports():
    current_app.logger.info("Reports page accessed")
    reports_list = IncidentReport.query.filter(IncidentReport.pdf_path.isnot(None)).order_by(IncidentReport.timestamp.desc()).all()
    return render_template('reports.html', reports=reports_list)

@dashboard_bp.route('/reports/generate', methods=['POST'])
@login_required
@role_required('admin')
def generate_report():
    incident = IncidentReport.query.filter_by(pdf_path=None).order_by(IncidentReport.timestamp.desc()).first()
    
    if incident:
        current_app.logger.info(f"Generating report for incident ID {incident.id}")
        explanation, recommendations = generate_incident_explanation(
            incident.vehicle_count, 
            incident.pedestrian_count, 
            incident.speed_kmh, 
            incident.distance_m, 
            incident.risk_level,
            global_traffic_state.get('congestion', 'Low')
        )
        incident.gemini_explanation = explanation
        incident.gemini_recommendations = recommendations
        db.session.commit()
        
        base_dir = os.path.dirname(os.path.dirname(current_app.root_path))
        reports_dir = os.path.join(base_dir, 'static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        filename = f"Incident_Report_{incident.id}_{incident.timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(reports_dir, filename)
        
        generate_pdf_report(incident, filepath)
        
        incident.pdf_path = filename
        db.session.commit()
        current_app.logger.info(f"Generated report {filename}")
        
    return redirect(url_for('dashboard.reports'))

@dashboard_bp.route('/reports/download/<int:report_id>')
@login_required
def download_report(report_id):
    incident = IncidentReport.query.get_or_404(report_id)
    if not incident.pdf_path:
        current_app.logger.warning(f"Download attempted for missing report PDF: ID {report_id}")
        return redirect(url_for('dashboard.reports'))
        
    base_dir = os.path.dirname(os.path.dirname(current_app.root_path))
    reports_dir = os.path.join(base_dir, 'static', 'reports')
    filepath = os.path.join(reports_dir, incident.pdf_path)
    
    current_app.logger.info(f"Downloading report {incident.pdf_path}")
    return send_file(filepath, as_attachment=True)

@dashboard_bp.route('/settings', methods=['GET', 'POST'])
@login_required
@role_required('admin')
def settings():
    if request.method == 'POST':
        speed_limit = request.form.get('speed_limit', '40')
        risk_sensitivity = request.form.get('risk_sensitivity', '75')
        email_alerts = 'true' if request.form.get('email_alerts') else 'false'
        sms_alerts = 'true' if request.form.get('sms_alerts') else 'false'
        enable_audio = 'true' if request.form.get('enable_audio') else 'false'
        audio_volume = request.form.get('audio_volume', '100')
        
        updates = {
            'speed_limit': speed_limit,
            'risk_sensitivity': risk_sensitivity,
            'email_alerts': email_alerts,
            'sms_alerts': sms_alerts,
            'enable_audio': enable_audio,
            'audio_volume': audio_volume
        }
        
        for key, value in updates.items():
            setting = Setting.query.filter_by(key=key).first()
            if setting:
                setting.value = value
            else:
                setting = Setting(key=key, value=value)
                db.session.add(setting)
        db.session.commit()
        current_app.logger.info("Settings updated")
        return redirect(url_for('dashboard.settings'))
        
    settings_records = Setting.query.all()
    settings_dict = {s.key: s.value for s in settings_records}
    current_app.logger.info("Settings page accessed")
    return render_template('settings.html', settings=settings_dict)

@dashboard_bp.route('/live-traffic')
@login_required
def live_traffic():
    google_maps_api_key = current_app.config.get('GOOGLE_MAPS_API_KEY', '')
    current_app.logger.info("Live traffic dashboard accessed")
    return render_template('live_traffic.html', google_maps_api_key=google_maps_api_key)

@dashboard_bp.route('/api/traffic-update', methods=['POST'])
@login_required
def update_traffic():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    global_traffic_state['congestion'] = data.get('congestion_level', 'Low')
    global_traffic_state['location'] = data.get('location', 'Central Hub')
    global_traffic_state['incidents'] = data.get('incident_count', 0)
    
    try:
        from ..models import TrafficSnapshot
        snapshot = TrafficSnapshot(
            location=global_traffic_state['location'],
            congestion_level=global_traffic_state['congestion'],
            travel_time_mins=data.get('travel_time_mins', 0),
            incident_count=global_traffic_state['incidents']
        )
        db.session.add(snapshot)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(f"Error saving traffic snapshot: {e}")
        
    return jsonify({'success': True, 'state': global_traffic_state})
