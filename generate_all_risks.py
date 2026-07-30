"""
Generate HIGH RISK collision demo videos for all cities.
Saves to: static/videos/<city>_risk.mp4
"""

import cv2
import numpy as np
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "static", "videos")
os.makedirs(OUT_DIR, exist_ok=True)

W, H   = 1280, 720
FPS    = 30
TOTAL_FRAMES = FPS * 18  # 18 seconds

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
WHITE        = (255, 255, 255)
BLACK        = (0,     0,   0)
YELLOW       = (0,   255, 255)
ORANGE       = (0,   165, 255)
SAFE_CLR     = (100, 220, 100)
MED_CLR      = (0,   220, 220)
HIGH_CLR     = (50,   50, 255)
CW_FILL      = (255,   0, 255)

# Crosswalk polygon (same formula as analytics_engine.py)
CW_PTS = np.array([
    [int(W * 0.15), int(H * 0.55)],
    [int(W * 0.85), int(H * 0.55)],
    [int(W * 0.90), int(H * 0.92)],
    [int(W * 0.10), int(H * 0.92)],
], np.int32).reshape((-1, 1, 2))


def draw_vehicle(frame, cx, cy, w, h, label, conf, color, speed_kmh=None, dist_m=None, direction=None, risk=None):
    x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 + 4, y2 + 4), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)

    if label in ("car", "motorcycle", "auto"):
        cv2.circle(frame, (x1 + 10, y2 - 6), 5, YELLOW, -1)
        cv2.circle(frame, (x2 - 10, y2 - 6), 5, YELLOW, -1)
    elif label in ("bus", "truck"):
        cv2.rectangle(frame, (x1 + 5, y1 + 5), (x1 + 30, y2 - 5), (20, 20, 25), -1)

    badge_txt = f"{label}: {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, badge_txt, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 1, cv2.LINE_AA)

    if speed_kmh is not None and dist_m is not None and direction:
        tag = f"{int(speed_kmh)}km/h | {dist_m:.1f}m | {direction}"
        cv2.putText(frame, tag, (x1, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1, cv2.LINE_AA)

    if risk and risk != "SAFE":
        rc = HIGH_CLR if risk == "HIGH RISK" else MED_CLR
        (rw, rh), _ = cv2.getTextSize(risk, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        rx1, ry1 = cx - rw // 2 - 6, cy - h // 2 - 40
        rx2, ry2 = cx + rw // 2 + 6, cy - h // 2 - 10
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), rc, -1)
        cv2.putText(frame, risk, (rx1 + 6, ry2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLACK, 2, cv2.LINE_AA)


def draw_pedestrian(frame, cx, cy, label="person", conf=0.94, risk=None):
    w, h = 34, 72
    x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    PED_CLR = (161, 227, 166)
    cv2.rectangle(frame, (x1, y1), (x2, y2), PED_CLR, 2)
    cv2.circle(frame, (cx, y1 - 10), 10, PED_CLR, 2)

    badge = f"{label}: {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), PED_CLR, -1)
    cv2.putText(frame, badge, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    if risk and risk != "SAFE":
        rc = HIGH_CLR if risk == "HIGH RISK" else MED_CLR
        cv2.putText(frame, risk, (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2, cv2.LINE_AA)


def render_risk_video(city_id, loc_name, road_col, threat_lbl, threat_col, threat_w, threat_h):
    path = os.path.join(OUT_DIR, f"{city_id}_risk.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    print(f"[*] Rendering {path} ...")
    
    for fi in range(TOTAL_FRAMES):
        t = fi / FPS
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        
        # 1. Background
        frame[:] = road_col
        cv2.rectangle(frame, (0, 0), (W, int(H * 0.18)), (38, 38, 42), -1)
        cv2.rectangle(frame, (0, int(H * 0.92)), (W, H), (38, 38, 42), -1)
        for lx in range(int(W * 0.2), int(W * 0.85), int(W * 0.15)):
            for seg_y in range(int(H * 0.18), int(H * 0.55), 40):
                cv2.rectangle(frame, (lx - 2, seg_y), (lx + 2, seg_y + 22), (55,55,65), -1)
            for seg_y in range(int(H * 0.55), int(H * 0.92), 40):
                cv2.rectangle(frame, (lx - 2, seg_y), (lx + 2, seg_y + 22), (55,55,65), -1)
        
        # 2. Crosswalk
        overlay = frame.copy()
        cv2.fillPoly(overlay, [CW_PTS], CW_FILL)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [CW_PTS], True, CW_FILL, 2)
        stripe_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(stripe_mask, [CW_PTS], 255)
        for sx in range(int(W * 0.15) + 20, int(W * 0.85), 55):
            roi = np.zeros_like(frame)
            cv2.rectangle(roi, (sx, int(H*0.55)), (sx + 28, int(H*0.92)), (190,190,195), -1)
            stripe_part = cv2.bitwise_and(roi, roi, mask=stripe_mask)
            cv2.addWeighted(stripe_part, 0.45, frame, 1.0, 0, frame)

        # 3. Trajectories
        if t < 4:
            car_cy = int(H * 0.10 + (H * 0.30) * (t / 4))
            car_speed = 35.0
        elif t < 8:
            car_cy = int(H * 0.40 + (H * 0.20) * ((t - 4) / 4))
            car_speed = 55.0
        elif t < 13:
            car_cy = int(H * 0.60 + (H * 0.20) * ((t - 8) / 5))
            car_speed = 72.0
        elif t < 14.5:
            car_cy = int(H * 0.80 + 10 * math.sin((t - 13) * 20))
            car_speed = max(0, 72 - 60 * (t - 13) / 1.5)
        else:
            car_cy = int(H * 0.80)
            car_speed = 0.0

        car_cx = int(W * 0.52)
        
        if t < 5:
            ped_cx = int(W * 0.14)
            ped_in_cw = False
        elif t < 14:
            progress = (t - 5) / 9
            ped_cx = int(W * 0.14 + W * 0.72 * progress)
            ped_in_cw = True
        else:
            ped_cx = int(W * 0.86)
            ped_in_cw = False
        ped_cy = int(H * 0.73)

        dist_px = math.sqrt((car_cx - ped_cx) ** 2 + (car_cy - ped_cy) ** 2)
        dist_m  = dist_px * 0.048
        
        risk = "SAFE"
        if dist_m < 4.0 and car_speed > 25 and ped_in_cw: risk = "HIGH RISK"
        elif dist_m < 9.0 and car_speed > 15 and ped_in_cw: risk = "MEDIUM"

        direction = "Approaching" if car_speed > 5 else "Stationary"

        # 4. Draw Vehicles & Peds
        draw_vehicle(frame, int(W*0.73), int(H*0.28), 80, 130, "bus", 0.91, (137,180,250))
        draw_vehicle(frame, int(W*0.32), int(H*0.42), 62, 110, "car", 0.88, (80,130,200))
        
        draw_vehicle(frame, car_cx, car_cy, threat_w, threat_h, threat_lbl, 0.97, threat_col,
                     speed_kmh=car_speed, dist_m=dist_m, direction=direction, risk=risk)
        
        draw_pedestrian(frame, ped_cx, ped_cy, conf=0.96, risk=risk)

        if ped_in_cw and dist_m < 40:
            color = HIGH_CLR if risk == "HIGH RISK" else (MED_CLR if risk == "MEDIUM" else SAFE_CLR)
            pts = np.linspace((car_cx,car_cy), (ped_cx,ped_cy), 20).astype(int)
            for i in range(0, len(pts) - 1, 2):
                cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), color, 2, cv2.LINE_AA)
            mid = ((car_cx + ped_cx) // 2, (car_cy + ped_cy) // 2)
            cv2.putText(frame, f"{dist_m:.1f}m", (mid[0], mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # 5. HUD & Overlays
        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (420, 150), (10, 10, 15), -1)
        cv2.addWeighted(hud, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"FPS: {int(28 + 3 * math.sin(t * 2))}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, SAFE_CLR, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Objects: 4", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ORANGE, 2, cv2.LINE_AA)
        
        sc = SAFE_CLR if not ped_in_cw else (255, 80, 80)
        cv2.putText(frame, f"Crosswalk: {'Occupied' if ped_in_cw else 'Empty'}", (12, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, sc, 2, cv2.LINE_AA)
        
        cv2.putText(frame, f"LIVE  T+{t:.1f}s", (W - 240, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, loc_name, (W - 400, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        if risk == "HIGH RISK":
            al = frame.copy()
            cv2.rectangle(al, (0, H - 60), (W, H), HIGH_CLR, -1)
            cv2.addWeighted(al, 0.85, frame, 0.15, 0, frame)
            msg = "!! COLLISION WARNING -- VEHICLE ENTERING OCCUPIED CROSSWALK !!"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, msg, ((W - tw) // 2, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

        risk_color = HIGH_CLR if risk == "HIGH RISK" else (MED_CLR if risk == "MEDIUM" else SAFE_CLR)
        badge = f"  RISK: {risk}  "
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        bx = (W - bw) // 2
        cv2.rectangle(frame, (bx - 6, 8), (bx + bw + 6, 8 + bh + 12), risk_color, -1)
        cv2.putText(frame, badge, (bx, 8 + bh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2, cv2.LINE_AA)

        if 13.0 <= t <= 13.4:
            flash = frame.copy()
            frame[:] = HIGH_CLR
            cv2.addWeighted(flash, 0.35, frame, 0.65, 0, frame)
            cv2.putText(frame, "COLLISION IMMINENT", (W // 2 - 280, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 4, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"  [OK] {path} saved.")

if __name__ == "__main__":
    CITIES = [
        ("hyderabad", "Hyderabad, India", (38,36,34), "truck", (100,100,150), 75, 145),
        ("seoul", "Seoul, South Korea", (32,32,38), "car", (200,200,200), 62, 110),
        ("bangkok", "Bangkok, Thailand", (40,38,30), "bus", (80,200,200), 75, 160),
        ("newyork", "New York, USA", (30,30,34), "car", (0,200,240), 62, 110),  # Yellow cab
        ("london", "London, UK", (35,33,32), "bus", (40,40,200), 80, 180),      # Red bus
        ("singapore", "Singapore", (30,32,36), "car", (180,180,180), 62, 110),
        ("tokyo", "Tokyo, Japan", (35,35,40), "truck", (200,200,200), 75, 150)
    ]
    for c_id, loc, rcol, t_lbl, t_col, t_w, t_h in CITIES:
        render_risk_video(c_id, loc, rcol, t_lbl, t_col, t_w, t_h)
