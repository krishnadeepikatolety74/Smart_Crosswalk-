"""
Generate a synthetic HIGH RISK collision demo video.
Saves to: static/videos/risk_demo.mp4
Run with: python generate_risk_demo.py
"""

import cv2
import numpy as np
import math
import os

# ── Output setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "static", "videos")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "risk_demo.mp4")

W, H   = 1280, 720
FPS    = 30
TOTAL_FRAMES = FPS * 18  # 18 seconds

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
BG_ROAD      = (28, 28, 32)
LANE_LINE    = (55, 55, 65)
CROSSWALK_BG = (50, 50, 55)
STRIPE_CLR   = (190, 190, 195)
CW_FILL      = (255,  0, 255)       # magenta crosswalk zone
PED_CLR      = (161, 227, 166)      # sage green
CAR_CLR      = (250, 180, 137)      # blue-ish
BUS_CLR      = (137, 180, 250)
SAFE_CLR     = (100, 220, 100)
MED_CLR      = (0,   220, 220)
HIGH_CLR     = (50,   50, 255)
WHITE        = (255, 255, 255)
ORANGE       = (0,   165, 255)
BLACK        = (0,     0,   0)
YELLOW       = (0,   255, 255)

# Crosswalk polygon (same formula as analytics_engine.py)
CW_PTS = np.array([
    [int(W * 0.15), int(H * 0.55)],
    [int(W * 0.85), int(H * 0.55)],
    [int(W * 0.90), int(H * 0.92)],
    [int(W * 0.10), int(H * 0.92)],
], np.int32).reshape((-1, 1, 2))


def draw_background(frame):
    """Road, lanes, buildings."""
    frame[:] = BG_ROAD

    # Sidewalks
    cv2.rectangle(frame, (0, 0), (W, int(H * 0.18)), (38, 38, 42), -1)
    cv2.rectangle(frame, (0, int(H * 0.92)), (W, H), (38, 38, 42), -1)

    # Lane dividers
    for lx in range(int(W * 0.2), int(W * 0.85), int(W * 0.15)):
        for seg_y in range(int(H * 0.18), int(H * 0.55), 40):
            cv2.rectangle(frame, (lx - 2, seg_y), (lx + 2, seg_y + 22), LANE_LINE, -1)
        for seg_y in range(int(H * 0.55), int(H * 0.92), 40):
            cv2.rectangle(frame, (lx - 2, seg_y), (lx + 2, seg_y + 22), LANE_LINE, -1)

    # Kerb lines
    cv2.line(frame, (0, int(H * 0.18)), (W, int(H * 0.18)), (70, 70, 80), 2)
    cv2.line(frame, (0, int(H * 0.92)), (W, int(H * 0.92)), (70, 70, 80), 2)

    # Simple building silhouettes (left)
    cv2.rectangle(frame, (0, 0), (110, int(H * 0.18)), (40, 40, 48), -1)
    cv2.rectangle(frame, (120, 0), (240, int(H * 0.14)), (45, 42, 52), -1)
    # Right side
    cv2.rectangle(frame, (W - 130, 0), (W, int(H * 0.18)), (40, 40, 48), -1)
    cv2.rectangle(frame, (W - 260, 0), (W - 140, int(H * 0.12)), (42, 45, 52), -1)


def draw_crosswalk(frame, alpha=0.18):
    """Draw magenta crosswalk zone exactly like analytics_engine.py."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [CW_PTS], CW_FILL)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [CW_PTS], True, CW_FILL, 2)

    # White zebra stripes inside
    cw_top    = int(H * 0.55)
    cw_bottom = int(H * 0.92)
    cw_left   = int(W * 0.15)
    cw_right  = int(W * 0.85)
    stripe_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(stripe_mask, [CW_PTS], 255)
    for sx in range(cw_left + 20, cw_right, 55):
        roi = np.zeros_like(frame)
        cv2.rectangle(roi, (sx, cw_top), (sx + 28, cw_bottom), STRIPE_CLR, -1)
        stripe_part = cv2.bitwise_and(roi, roi, mask=stripe_mask)
        cv2.addWeighted(stripe_part, 0.45, frame, 1.0, 0, frame)


def draw_vehicle(frame, cx, cy, w, h, label, conf, color, speed_kmh=None,
                 dist_m=None, direction=None, risk=None):
    x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2

    # Shadow
    cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 + 4, y2 + 4), (0, 0, 0), -1)

    # Body
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 2)

    # Headlights / wheels feel
    if label in ("car", "motorcycle"):
        cv2.circle(frame, (x1 + 10, y2 - 6), 5, YELLOW, -1)
        cv2.circle(frame, (x2 - 10, y2 - 6), 5, YELLOW, -1)
    elif label == "bus":
        cv2.rectangle(frame, (x1 + 5, y1 + 5), (x1 + 30, y2 - 5), (20, 20, 25), -1)

    # Label badge
    badge_txt = f"{label}: {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, badge_txt, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    # Speed / distance / direction tag (like real YOLO overlay)
    if speed_kmh is not None and dist_m is not None and direction:
        tag = f"{int(speed_kmh)}km/h | {dist_m:.1f}m | {direction}"
        cv2.putText(frame, tag, (x1, y1 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, YELLOW, 1, cv2.LINE_AA)

    # Risk badge on vehicle
    if risk and risk != "SAFE":
        rc = HIGH_CLR if risk == "HIGH RISK" else MED_CLR
        (rw, rh), _ = cv2.getTextSize(risk, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        rx1, ry1 = cx - rw // 2 - 6, cy - h // 2 - 40
        rx2, ry2 = cx + rw // 2 + 6, cy - h // 2 - 10
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), rc, -1)
        cv2.putText(frame, risk, (rx1 + 6, ry2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, BLACK, 2, cv2.LINE_AA)

    return (x1, y1, x2, y2)


def draw_pedestrian(frame, cx, cy, label="person", conf=0.94, risk=None):
    w, h = 34, 72
    x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2

    # Body
    cv2.rectangle(frame, (x1, y1), (x2, y2), PED_CLR, 2)

    # Head circle
    cv2.circle(frame, (cx, y1 - 10), 10, PED_CLR, 2)

    # Label
    badge = f"{label}: {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), PED_CLR, -1)
    cv2.putText(frame, badge, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)

    if risk and risk != "SAFE":
        rc = HIGH_CLR if risk == "HIGH RISK" else MED_CLR
        cv2.putText(frame, risk, (x1, y1 - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2, cv2.LINE_AA)

    return (cx, cy)


def draw_distance_line(frame, p1, p2, dist_m, risk):
    color = HIGH_CLR if risk == "HIGH RISK" else (MED_CLR if risk == "MEDIUM" else SAFE_CLR)
    # Dashed line
    pts = np.linspace(p1, p2, 20).astype(int)
    for i in range(0, len(pts) - 1, 2):
        cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), color, 2, cv2.LINE_AA)

    # Distance label at midpoint
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    (tw, th), _ = cv2.getTextSize(f"{dist_m:.1f}m", cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (mid[0] - 4, mid[1] - th - 4),
                  (mid[0] + tw + 4, mid[1] + 4), BLACK, -1)
    cv2.putText(frame, f"{dist_m:.1f}m", (mid[0], mid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_hud(frame, t, fps_val, ped_count, status, risk, alert_text=None):
    """Top-left HUD — matches real dashboard overlays."""
    # Semi-transparent HUD bar
    hud = frame.copy()
    cv2.rectangle(hud, (0, 0), (420, 150), (10, 10, 15), -1)
    cv2.addWeighted(hud, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"FPS: {int(fps_val)}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, SAFE_CLR, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {ped_count + 3}", (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ORANGE, 2, cv2.LINE_AA)

    sc = SAFE_CLR if status == "Crosswalk Empty" else (255, 80, 80)
    cv2.putText(frame, f"Crosswalk: {status}", (12, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, sc, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Pedestrians: {ped_count}", (12, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1, cv2.LINE_AA)

    # Top-right: timestamp + mode badge
    ts = f"LIVE  T+{t:.1f}s"
    cv2.putText(frame, ts, (W - 240, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "YOLOv8 | ByteTrack", (W - 240, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 160), 1, cv2.LINE_AA)

    # Alert banner
    if alert_text:
        al = frame.copy()
        cv2.rectangle(al, (0, H - 60), (W, H), HIGH_CLR, -1)
        cv2.addWeighted(al, 0.85, frame, 0.15, 0, frame)
        (tw, _), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(frame, alert_text, ((W - tw) // 2, H - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

    # Risk badge top-centre
    risk_color = HIGH_CLR if risk == "HIGH RISK" else (MED_CLR if risk == "MEDIUM" else SAFE_CLR)
    badge = f"  RISK: {risk}  "
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    bx = (W - bw) // 2
    cv2.rectangle(frame, (bx - 6, 8), (bx + bw + 6, 8 + bh + 12), risk_color, -1)
    cv2.putText(frame, badge, (bx, 8 + bh + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2, cv2.LINE_AA)


# ── Scene script ──────────────────────────────────────────────────────────────
# Phase 1 (0-4s):   SAFE — car far away, ped waiting on kerb
# Phase 2 (4-8s):   MEDIUM — car approaches, ped steps in
# Phase 3 (8-13s):  HIGH RISK — car accelerates, ped mid-crossing, collision imm.
# Phase 4 (13-16s): COLLISION FLASH + brake, car screeches to halt
# Phase 5 (16-18s): Recovery — car stopped, ped safe, risk back to SAFE

print("[*] Generating risk demo video ...")
for fi in range(TOTAL_FRAMES):
    t = fi / FPS

    frame = np.zeros((H, W, 3), dtype=np.uint8)
    draw_background(frame)
    draw_crosswalk(frame)

    # ── Positions ────────────────────────────────────────────────────────────
    # Main threat car (comes from top, moves down)
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
        car_cy = int(H * 0.80 + 10 * math.sin((t - 13) * 20))  # braking shake
        car_speed = max(0, 72 - 60 * (t - 13) / 1.5)
    else:
        car_cy = int(H * 0.80)
        car_speed = 0.0

    car_cx = int(W * 0.52)

    # Secondary car (parked/slow on left lane)
    car2_cx = int(W * 0.32)
    car2_cy = int(H * 0.42)

    # Bus in right lane (stationary)
    bus_cx = int(W * 0.73)
    bus_cy = int(H * 0.28)

    # Pedestrian (walks right across crosswalk)
    if t < 5:
        ped_cx = int(W * 0.14)
        ped_cy = int(H * 0.73)
        ped_in_cw = False
    elif t < 14:
        progress = (t - 5) / 9
        ped_cx = int(W * 0.14 + W * 0.72 * progress)
        ped_cy = int(H * 0.73)
        ped_in_cw = True
    else:
        ped_cx = int(W * 0.86)
        ped_cy = int(H * 0.73)
        ped_in_cw = False

    # Second pedestrian (child, smaller — starts later)
    if t > 7:
        ped2_cx = int(W * 0.22 + W * 0.40 * min(1.0, (t - 7) / 8))
        ped2_cy = int(H * 0.68)
        show_ped2 = True
    else:
        show_ped2 = False

    # Distance (car centroid → ped centroid)
    dist_px = math.sqrt((car_cx - ped_cx) ** 2 + (car_cy - ped_cy) ** 2)
    dist_m  = dist_px * 0.048   # pixel-to-metre calibration

    # Risk level
    if dist_m < 4.0 and car_speed > 25 and ped_in_cw:
        risk = "HIGH RISK"
    elif dist_m < 9.0 and car_speed > 15 and ped_in_cw:
        risk = "MEDIUM"
    else:
        risk = "SAFE"

    # Status
    status = "Crosswalk Occupied" if ped_in_cw else "Crosswalk Empty"
    ped_count = (1 if ped_in_cw else 0) + (1 if show_ped2 else 0)
    direction = "Approaching" if car_speed > 5 else "Stationary"

    # Alert banner text
    alert_text = None
    if risk == "HIGH RISK":
        alert_text = "!! COLLISION WARNING -- VEHICLE ENTERING OCCUPIED CROSSWALK !!"
    elif risk == "MEDIUM":
        alert_text = "CAUTION -- Vehicle approaching pedestrian zone"
    if car_speed > 60:
        alert_text = "OVERSPEED ALERT -- Vehicle at {:.0f} km/h in pedestrian zone".format(car_speed)

    # ── Draw vehicles ─────────────────────────────────────────────────────────
    # Bus (background, draw first)
    draw_vehicle(frame, bus_cx, bus_cy, 80, 130, "bus", 0.91, BUS_CLR)

    # Parked / slow car
    draw_vehicle(frame, car2_cx, car2_cy, 62, 110, "car", 0.88, (80, 130, 200))

    # Main threat car
    draw_vehicle(frame, car_cx, car_cy, 70, 120, "car", 0.97, CAR_CLR,
                 speed_kmh=car_speed, dist_m=dist_m,
                 direction=direction, risk=risk)

    # Motorcycle on far right lane
    mc_cx = int(W * 0.82)
    mc_cy = int(H * 0.35 + H * 0.10 * (t / 18))
    draw_vehicle(frame, mc_cx, mc_cy, 26, 52, "motorcycle", 0.84, (200, 100, 230),
                 speed_kmh=28.0)

    # ── Draw pedestrians ──────────────────────────────────────────────────────
    if show_ped2:
        draw_pedestrian(frame, ped2_cx, ped2_cy, conf=0.90)
    draw_pedestrian(frame, ped_cx, ped_cy, conf=0.96, risk=risk)

    # ── Distance line ─────────────────────────────────────────────────────────
    if ped_in_cw and dist_m < 40:
        draw_distance_line(frame,
                           (car_cx, car_cy),
                           (ped_cx, ped_cy),
                           dist_m, risk)

    # ── Traffic light (top-right) ─────────────────────────────────────────────
    tl_x, tl_y = W - 60, 80
    cv2.rectangle(frame, (tl_x - 18, tl_y - 60), (tl_x + 18, tl_y + 62), (20, 20, 20), -1)
    cv2.rectangle(frame, (tl_x - 18, tl_y - 60), (tl_x + 18, tl_y + 62), (60, 60, 60), 2)
    # Red on during approach
    tl_red   = (0, 0, 220) if t < 14 else (20, 20, 60)
    tl_yel   = (0, 220, 220) if (13 <= t < 14) else (20, 60, 20)
    tl_green = (0, 200, 60) if t >= 14 else (20, 50, 20)
    cv2.circle(frame, (tl_x, tl_y - 38), 12, tl_red,   -1)
    cv2.circle(frame, (tl_x, tl_y      ), 12, tl_yel,   -1)
    cv2.circle(frame, (tl_x, tl_y + 38), 12, tl_green, -1)

    # Signal overlay (like analytics_engine.py)
    sig_state = "Red" if t < 13 else ("Yellow" if t < 14 else "Green")
    sig_color = (0, 0, 255) if sig_state == "Red" else ((0, 255, 255) if sig_state == "Yellow" else (0, 255, 0))
    cv2.putText(frame, f"Signal: {sig_state}  Drv: {'STOP' if sig_state == 'Red' else 'GO'}",
                (tl_x - 180, tl_y - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, sig_color, 1, cv2.LINE_AA)

    # ── HUD ──────────────────────────────────────────────────────────────────
    fps_sim = 28 + 3 * math.sin(t * 2)   # simulate realistic FPS jitter
    draw_hud(frame, t, fps_sim, ped_count, status, risk, alert_text)

    # Collision flash frame (red flash)
    if 13.0 <= t <= 13.4:
        flash = frame.copy()
        frame[:] = HIGH_CLR
        cv2.addWeighted(flash, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, "COLLISION IMMINENT", (W // 2 - 280, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 4, cv2.LINE_AA)

    writer.write(frame)

writer.release()
print(f"[OK] Video saved -> {OUT_PATH}")
print(f"     Size: {os.path.getsize(OUT_PATH) // 1024} KB")
print()
print("[INFO] In the browser: go to Live Camera -> click the feed button for risk_demo")
print("       The file is at:  static/videos/risk_demo.mp4")
