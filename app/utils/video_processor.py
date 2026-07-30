import cv2
import logging
import time
from flask import current_app
from .analytics_engine import AnalyticsEngine
from .model_loader import get_yolo_model

TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 9]

def process_video(input_path, output_path):
    yolo_model = get_yolo_model()
    if not yolo_model:
        return {"error": "YOLO model not loaded", "count": 0, "detections": []}

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {"error": "Could not open video", "count": 0, "detections": []}
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # mp4v codec for MP4 output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        max_counts = {}
        sum_confs = {}
        obs_counts = {}
        prev_time = 0
        engine = AnalyticsEngine(fps=fps)
        
        peak_frame = None
        max_seen_objects = -1
        
        vehicle_timeline = []
        risk_timeline = []
        
        app_obj = current_app._get_current_object() if current_app else None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            # Apply analytics on the raw frame (matching the live camera's custom YOLO styling)
            plotted_frame = engine.process_frame(frame, results, app=app_obj)
            
            # Calculate object count for this frame
            frame_counts = {}
            veh_count = 0
            peds = []
            vehs = []
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                conf = float(box.conf[0])
                frame_counts[class_name] = frame_counts.get(class_name, 0) + 1
                sum_confs[class_name] = sum_confs.get(class_name, 0.0) + conf
                obs_counts[class_name] = obs_counts.get(class_name, 0) + 1
                
                # Identify pedestrians and vehicles for distance-based risk timelines
                if class_id == 0: # person
                    xyxy = box.xyxy[0].tolist()
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    peds.append((cx, cy))
                elif class_id in [1, 2, 3, 5, 7]: # vehicle classes
                    veh_count += 1
                    xyxy = box.xyxy[0].tolist()
                    cx = int((xyxy[0] + xyxy[2]) / 2)
                    cy = int((xyxy[1] + xyxy[3]) / 2)
                    vehs.append((cx, cy))
                    
            # Compute frame risk score based on minimum distance
            frame_risk = 0
            if peds and vehs:
                min_dist_px = 999999
                for px, py in peds:
                    for vx, vy in vehs:
                        dist = ((px - vx)**2 + (py - vy)**2)**0.5
                        if dist < min_dist_px:
                            min_dist_px = dist
                # Approximate scaling
                min_dist_m = min_dist_px * 0.04
                if min_dist_m < 3.5:
                    frame_risk = 85 # high risk %
                elif min_dist_m < 6.5:
                    frame_risk = 45 # medium risk %
                elif min_dist_m < 15.0:
                    frame_risk = 20 # low risk %
            
            vehicle_timeline.append(veh_count)
            risk_timeline.append(frame_risk)
                
            # Update max counts
            for cls, count in frame_counts.items():
                max_counts[cls] = max(max_counts.get(cls, 0), count)
                
            object_count = len(results[0].boxes)
            
            # Keep track of peak detection frame
            if object_count > max_seen_objects:
                max_seen_objects = object_count
                peak_frame = plotted_frame.copy()
            
            # Calculate Processing FPS
            curr_time = time.time()
            processing_fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            out.write(plotted_frame)
            
        cap.release()
        out.release()
        
        # Save peak detection frame snapshot
        if peak_frame is not None:
            snapshot_path = output_path.replace('.mp4', '_snapshot.jpg')
            cv2.imwrite(snapshot_path, peak_frame)
            
        # Downsample timelines to 60 data points for UI line charts
        target_size = 60
        downsampled_vehicles = []
        downsampled_risk = []
        if len(vehicle_timeline) > 0:
            for i in range(target_size):
                idx = int(i * len(vehicle_timeline) / target_size)
                downsampled_vehicles.append(vehicle_timeline[idx])
                downsampled_risk.append(risk_timeline[idx])
        else:
            downsampled_vehicles = [0] * target_size
            downsampled_risk = [0] * target_size
        
        detections = []
        for cls, count in max_counts.items():
            avg_conf = sum_confs[cls] / obs_counts[cls] if obs_counts.get(cls, 0) > 0 else 0
            detections.append({"class_name": cls, "confidence": avg_conf, "count": count})
        total_objects = sum(max_counts.values())
        
        return {
            "success": True,
            "count": total_objects,
            "detections": detections,
            "snapshot_url": output_path.split('static')[-1].replace('\\', '/').replace('.mp4', '_snapshot.jpg'),
            "vehicle_trend": downsampled_vehicles,
            "risk_trend": downsampled_risk
        }
    except Exception as e:
        logging.error(f"Error processing video {input_path}: {e}")
        return {"error": str(e), "count": 0, "detections": []}
