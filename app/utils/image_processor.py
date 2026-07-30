import cv2
import logging
from .model_loader import get_yolo_model

# TARGET CLASSES (COCO dataset)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 9]

def process_image(image_path, output_path):
    yolo_model = get_yolo_model()
    if not yolo_model:
        return {"error": "YOLO model not loaded", "count": 0, "detections": []}

    try:
        # Run YOLO inference on all classes
        results = yolo_model(image_path, verbose=False)
        
        result = results[0]
        
        # Save plotted image
        plotted_img = result.plot()
        cv2.imwrite(output_path, plotted_img)
        
        detections = []
        vehicles = []
        pedestrians = []
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            detections.append({
                "class_name": class_name,
                "confidence": conf
            })
            
            if class_name == 'person':
                pedestrians.append((cx, cy))
            elif class_name in ['car', 'bus', 'truck', 'motorcycle']:
                vehicles.append((cx, cy))
                
        # Risk Calculation
        import math
        min_dist_m = 999.0
        risk_level = "SAFE"
        pixel_to_meter = 0.05
        
        if pedestrians and vehicles:
            for v_cx, v_cy in vehicles:
                for p_cx, p_cy in pedestrians:
                    dist_px = math.sqrt((v_cx - p_cx)**2 + (v_cy - p_cy)**2)
                    dist_m = dist_px * pixel_to_meter
                    if dist_m < min_dist_m:
                        min_dist_m = dist_m
            
            if min_dist_m < 3.0:
                risk_level = "HIGH RISK"
            elif min_dist_m < 5.0:
                risk_level = "MEDIUM"
        else:
            min_dist_m = 0.0
            
        # Gemini Recommendation
        from .gemini_helper import generate_incident_explanation
        from .analytics_engine import global_traffic_state
        explanation, recommendations = generate_incident_explanation(
            len(vehicles), len(pedestrians), 0.0, min_dist_m, risk_level, global_traffic_state.get('congestion', 'Low')
        )
            
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "risk_level": risk_level,
            "gemini_explanation": explanation,
            "gemini_recommendation": recommendations,
            "vehicle_count": len(vehicles),
            "pedestrian_count": len(pedestrians),
            "min_dist_m": min_dist_m
        }
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return {"error": str(e), "count": 0, "detections": []}
