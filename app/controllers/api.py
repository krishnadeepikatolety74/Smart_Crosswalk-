from flask import Blueprint, request, jsonify, current_app
import google.generativeai as genai
from ..utils.helpers import login_required
from ..models import Alert
from ..utils.analytics_engine import global_crosswalk_state

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/chat', methods=['POST'])
@login_required
def chat():
    current_app.logger.info("API chat endpoint accessed")
    gemini_api_key = current_app.config.get('GEMINI_API_KEY')
    if not gemini_api_key:
        return jsonify({"response": "Gemini API Key is not set. Please set the GEMINI_API_KEY environment variable."})
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    
    user_message = request.json.get('message', '')
    
    # Gather real detection data
    from ..models import IncidentReport
    
    recent_incidents = IncidentReport.query.order_by(IncidentReport.timestamp.desc()).limit(5).all()
    context_str = f"Current Crosswalk Status: {global_crosswalk_state.get('status', 'SAFE')}, Pedestrians: {global_crosswalk_state.get('pedestrian_count', 0)}\n"
    if recent_incidents:
        context_str += "Recent Incidents (Real Data):\n"
        for inc in recent_incidents:
            context_str += f"- {inc.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: {inc.vehicle_count} vehicles, {inc.pedestrian_count} pedestrians, Speed: {inc.speed_kmh:.1f} km/h, Risk: {inc.risk_level}. Detected: {inc.detected_objects}\n"
    
    prompt = f"""
    You are an expert AI Road Safety Assistant for a Smart Crosswalk System. 
    Your core responsibilities:
    1. Explain detections.
    2. Predict safety risks.
    3. Generate AI Incident Reports summaries if requested.
    4. Recommend actions based on real data.
    5. Answer road safety questions.
    
    You MUST rely exclusively on the Real-Time System Data Context provided below. DO NOT invent dummy responses or fictional scenarios. 
    
    System Data Context:
    {context_str}
    
    User Question: {user_message}
    """
    
    try:
        response = model.generate_content(prompt)
        return jsonify({"response": response.text})
    except Exception as e:
        current_app.logger.error(f"Error communicating with AI: {str(e)}")
        return jsonify({"response": f"Error communicating with AI: {str(e)}"})

@api_bp.route('/status')
def api_status():
    recent_alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(5).all()
    alerts_data = [{"type": a.alert_type, "desc": a.description, "time": a.timestamp.strftime("%H:%M:%S")} for a in recent_alerts]
    
    return jsonify({
        "status": global_crosswalk_state["status"],
        "count": global_crosswalk_state["pedestrian_count"],
        "alerts": alerts_data,
        "counts": global_crosswalk_state.get("counts", {"person": 0, "car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "bicycle": 0}),
        "fps": global_crosswalk_state.get("fps", 30),
        "inference_ms": global_crosswalk_state.get("inference_ms", 12),
        "risk_level": global_crosswalk_state.get("risk_level", "SAFE"),
        "traffic_density": global_crosswalk_state.get("traffic_density", "Light")
    })
