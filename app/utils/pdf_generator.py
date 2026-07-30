from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def generate_pdf_report(report_data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1 * inch, height - 1 * inch, "Smart Crosswalk Incident Report")
    
    # Data
    c.setFont("Helvetica", 12)
    y_position = height - 1.5 * inch
    
    date_str = report_data.timestamp.strftime('%Y-%m-%d')
    time_str = report_data.timestamp.strftime('%H:%M:%S')
    
    lines = [
        f"Report ID: {report_data.id}",
        f"Date: {date_str}",
        f"Time: {time_str}",
        f"Vehicle Count: {report_data.vehicle_count}",
        f"Pedestrian Count: {report_data.pedestrian_count}",
        f"Speed: {report_data.speed_kmh:.1f} km/h",
        f"Distance: {report_data.distance_m:.1f} m",
        f"Risk Level: {report_data.risk_level}",
        "",
        "Gemini Explanation:",
    ]
    
    for line in lines:
        c.drawString(1 * inch, y_position, line)
        y_position -= 0.3 * inch
        
    import textwrap
    
    # Handle Explanation
    c.setFont("Helvetica-Oblique", 11)
    explanation = report_data.gemini_explanation or "No explanation provided."
    wrapped_text = textwrap.wrap(explanation, width=80)
    for line in wrapped_text:
        c.drawString(1 * inch, y_position, line)
        y_position -= 0.25 * inch
        
    y_position -= 0.1 * inch
    
    # Handle Recommendations
    c.setFont("Helvetica", 12)
    c.drawString(1 * inch, y_position, "Recommendations:")
    y_position -= 0.3 * inch
    
    c.setFont("Helvetica-Oblique", 11)
    recommendations = report_data.gemini_recommendations or "No recommendations provided."
    wrapped_rec = textwrap.wrap(recommendations, width=80)
    for line in wrapped_rec:
        c.drawString(1 * inch, y_position, line)
        y_position -= 0.25 * inch
        
    c.save()
    return output_path
