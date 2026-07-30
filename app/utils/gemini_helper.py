import os
import google.generativeai as genai

def generate_incident_explanation(vehicle_count, ped_count, speed, distance, risk, traffic_congestion="Low"):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (f"Gemini API Key missing - AI analysis disabled. Rule-based explanation: A {risk} risk event occurred involving {vehicle_count} vehicles and {ped_count} pedestrians at a speed of {speed:.1f} km/h with a distance of {distance:.1f} meters. Traffic is {traffic_congestion}.", "No recommendations available.")
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')
        prompt = f"""
        You are an AI Traffic Safety Analyst.
        An incident occurred with the following parameters:
        - Vehicles present: {vehicle_count}
        - Pedestrians present: {ped_count}
        - Vehicle Speed: {speed:.1f} km/h
        - Distance to Pedestrian: {distance:.1f} m
        - Evaluated Risk: {risk}
        - Traffic Congestion Level: {traffic_congestion}
        
        Provide the output in exactly two lines:
        Explanation: [Concise 2-sentence explanation of why this was flagged as {risk} and the primary danger]
        Recommendations: [Concise 2-sentence actionable safety recommendations]
        """
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        explanation = "Error parsing explanation"
        recommendations = "Error parsing recommendations"
        
        for line in text.split('\n'):
            if line.startswith('Explanation:'):
                explanation = line.replace('Explanation:', '').strip()
            elif line.startswith('Recommendations:'):
                recommendations = line.replace('Recommendations:', '').strip()
                
        return (explanation, recommendations)
    except Exception as e:
        return (f"Gemini AI Analysis Error: {str(e)}", "Error")
