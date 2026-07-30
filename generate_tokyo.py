# -*- coding: utf-8 -*-
"""
Generate synthetic Tokyo Shibuya-style scramble crossing video.
Saves to: static/videos/tokyo.mp4
Run: python generate_tokyo.py
"""
import cv2
import numpy as np
import math
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "static", "videos")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "tokyo.mp4")

W, H   = 1280, 720
FPS    = 30
SECS   = 30
TOTAL  = FPS * SECS

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

# --- Colour palette (BGR) ---
ASPHALT   = (38, 38, 42)
SIDEWALK  = (70, 68, 65)
LANE_MRK  = (160, 160, 165)
ZEBRA_CLR = (200, 198, 195)
CW_FILL   = (255,   0, 255)
PED_COLS  = [(161,227,166),(137,180,250),(250,179,135),(243,139,168),(205,214,244)]
CAR_COLS  = [(80,130,200),(130,100,200),(200,110,80),(80,200,130),(180,180,80)]
WHITE     = (255,255,255)
BLACK     = (0,0,0)
YELLOW    = (0,255,255)
GREEN_SIG = (0,220,60)
RED_SIG   = (0,0,220)
ORANGE    = (0,165,255)
SAFE_CLR  = (60,220,60)
HIGH_CLR  = (50,50,255)

# Intersection centre
CX, CY = W//2, H//2
# Crosswalk half-widths
CW_W = 200   # perpendicular width of each crosswalk strip
CW_H =  90   # length along road direction

# Crosswalk rectangles: (x1,y1,x2,y2) — one on each side
# North crosswalk (top)
NC = (CX - CW_W, CY - 180 - CW_H,  CX + CW_W, CY - 180)
# South crosswalk (bottom)
SC = (CX - CW_W, CY + 180,          CX + CW_W, CY + 180 + CW_H)
# West crosswalk (left)
WC = (CX - 180 - CW_H, CY - CW_W,  CX - 180,  CY + CW_W)
# East crosswalk (right)
EC = (CX + 180,          CY - CW_W, CX + 180 + CW_H, CY + CW_W)

# All 4 crosswalk polygons as poly points (for pink overlay)
def rect_poly(r):
    x1,y1,x2,y2 = r
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.int32).reshape(-1,1,2)

ALL_CW_POLYS = [rect_poly(r) for r in [NC, SC, WC, EC]]


def draw_road(frame):
    frame[:] = ASPHALT
    # Sidewalk corners
    cv2.rectangle(frame, (0, 0),       (CX-180, CY-180), SIDEWALK, -1)
    cv2.rectangle(frame, (CX+180, 0),  (W, CY-180),      SIDEWALK, -1)
    cv2.rectangle(frame, (0, CY+180),  (CX-180, H),      SIDEWALK, -1)
    cv2.rectangle(frame, (CX+180,CY+180),(W, H),          SIDEWALK, -1)
    # Centre intersection box (lighter)
    cv2.rectangle(frame, (CX-180,CY-180),(CX+180,CY+180),(45,45,50),-1)
    # Lane markings — horizontal road
    for y_lane in range(CY-130, CY+131, 65):
        for sx in range(0, CX-180, 60):
            cv2.rectangle(frame,(sx,y_lane-2),(sx+38,y_lane+2),LANE_MRK,-1)
        for sx in range(CX+180, W, 60):
            cv2.rectangle(frame,(sx,y_lane-2),(sx+38,y_lane+2),LANE_MRK,-1)
    # Lane markings — vertical road
    for x_lane in range(CX-130, CX+131, 65):
        for sy in range(0, CY-180, 60):
            cv2.rectangle(frame,(x_lane-2,sy),(x_lane+2,sy+38),LANE_MRK,-1)
        for sy in range(CY+180, H, 60):
            cv2.rectangle(frame,(x_lane-2,sy),(x_lane+2,sy+38),LANE_MRK,-1)


def draw_crosswalks(frame, alpha=0.15):
    # Zebra stripes
    for r in [NC, SC]:
        x1,y1,x2,y2 = r
        for sx in range(x1+10, x2-10, 32):
            cv2.rectangle(frame,(sx,y1),(sx+18,y2),ZEBRA_CLR,-1)
    for r in [WC, EC]:
        x1,y1,x2,y2 = r
        for sy in range(y1+10, y2-10, 32):
            cv2.rectangle(frame,(x1,sy),(x2,sy+18),ZEBRA_CLR,-1)
    # Pink overlay
    ov = frame.copy()
    for poly in ALL_CW_POLYS:
        cv2.fillPoly(ov, [poly], CW_FILL)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)
    for poly in ALL_CW_POLYS:
        cv2.polylines(frame, [poly], True, CW_FILL, 2)


def in_any_crosswalk(cx, cy):
    for r in [NC, SC, WC, EC]:
        x1,y1,x2,y2 = r
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


class Pedestrian:
    _id = 0
    def __init__(self, rng):
        Pedestrian._id += 1
        self.id   = Pedestrian._id
        self.col  = rng.choice(PED_COLS)
        self.conf = round(0.88 + rng.random()*0.09, 2)
        self.w, self.h_box = 22, 48
        # Choose crossing direction: 0=N->S, 1=S->N, 2=W->E, 3=E->W, diagonals for scramble
        self.dir = rng.randint(0, 5)
        speeds = [1.2, 1.4, 1.1, 1.5, 1.3, 1.2]
        self.spd = speeds[self.dir] + rng.random()*0.3
        # Start positions & targets
        if self.dir == 0:   # N->S
            self.x = rng.uniform(CX-160, CX+160)
            self.y = float(NC[1] - 10)
            self.tx, self.ty = self.x + rng.uniform(-20,20), float(SC[3] + 10)
        elif self.dir == 1: # S->N
            self.x = rng.uniform(CX-160, CX+160)
            self.y = float(SC[3] + 10)
            self.tx, self.ty = self.x + rng.uniform(-20,20), float(NC[1] - 10)
        elif self.dir == 2: # W->E
            self.x = float(WC[0] - 10)
            self.y = rng.uniform(CY-160, CY+160)
            self.tx, self.ty = float(EC[2] + 10), self.y + rng.uniform(-20,20)
        elif self.dir == 3: # E->W
            self.x = float(EC[2] + 10)
            self.y = rng.uniform(CY-160, CY+160)
            self.tx, self.ty = float(WC[0] - 10), self.y + rng.uniform(-20,20)
        elif self.dir == 4: # NW->SE diagonal
            self.x = float(WC[0] - 10)
            self.y = float(NC[1] - 10)
            self.tx, self.ty = float(EC[2]+10), float(SC[3]+10)
        else:               # NE->SW diagonal
            self.x = float(EC[2] + 10)
            self.y = float(NC[1] - 10)
            self.tx, self.ty = float(WC[0]-10), float(SC[3]+10)
        self.done = False

    def step(self):
        dx = self.tx - self.x
        dy = self.ty - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 3:
            self.done = True
            return
        self.x += self.spd * dx / dist
        self.y += self.spd * dy / dist

    def draw(self, frame, risk="SAFE"):
        cx, cy = int(self.x), int(self.y)
        x1,y1 = cx - self.w//2, cy - self.h_box//2
        x2,y2 = cx + self.w//2, cy + self.h_box//2
        # Body
        cv2.rectangle(frame,(x1,y1),(x2,y2),self.col,2)
        # Head
        cv2.circle(frame,(cx,y1-8),8,self.col,2)
        # Label
        lbl = f"person:{self.conf:.2f}"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
        cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),self.col,-1)
        cv2.putText(frame,lbl,(x1+2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.38,BLACK,1,cv2.LINE_AA)


class Vehicle:
    _id = 100
    def __init__(self, lane, direction, rng):
        Vehicle._id += 1
        self.id  = Vehicle._id
        self.col = rng.choice(CAR_COLS)
        self.conf= round(0.85 + rng.random()*0.12, 2)
        self.lane = lane       # 0=left, 1=right
        self.direction = direction  # 'N','S','E','W'
        self.speed = rng.uniform(1.5, 2.5)
        self.waiting = True
        self.w, self.h_box = 55, 100
        self.spd_kmh = rng.uniform(20, 45)
        if direction == 'S':
            self.x = CX - 90 + lane*90
            self.y = -60.0
            self.tx, self.ty = float(self.x), float(H + 60)
        elif direction == 'N':
            self.x = CX + 90 - lane*90
            self.y = float(H + 60)
            self.tx, self.ty = float(self.x), -60.0
        elif direction == 'E':
            self.x = -60.0
            self.y = CY - 80 + lane*90
            self.tx, self.ty = float(W + 60), float(self.y)
        else:  # W
            self.x = float(W + 60)
            self.y = CY + 80 - lane*90
            self.tx, self.ty = -60.0, float(self.y)
        self.done = False

    def step(self, green):
        # Vehicles move only when green light
        if not green:
            self.waiting = True
            return
        self.waiting = False
        dx = self.tx - self.x
        dy = self.ty - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 3:
            self.done = True
            return
        self.x += self.speed * dx / dist
        self.y += self.speed * dy / dist

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        if self.direction in ('N','S'):
            w2, h2 = self.w//2, self.h_box//2
        else:
            w2, h2 = self.h_box//2, self.w//2
        x1,y1,x2,y2 = cx-w2, cy-h2, cx+w2, cy+h2
        cv2.rectangle(frame,(x1+3,y1+3),(x2+3,y2+3),(0,0,0),-1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),self.col,-1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),WHITE,2)
        lbl = f"car:{self.conf:.2f}"
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
        cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+4,y1),self.col,-1)
        cv2.putText(frame,lbl,(x1+2,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.4,BLACK,1,cv2.LINE_AA)
        # Speed tag
        tag = f"{int(self.spd_kmh)}km/h"
        cv2.putText(frame,tag,(x1,y1-14),cv2.FONT_HERSHEY_SIMPLEX,0.38,YELLOW,1,cv2.LINE_AA)


def draw_traffic_lights(frame, green_for_peds):
    """Four traffic lights at each corner of the intersection."""
    corners = [
        (CX-185, CY-185),
        (CX+165, CY-185),
        (CX-185, CY+160),
        (CX+165, CY+160),
    ]
    for (lx,ly) in corners:
        cv2.rectangle(frame,(lx-10,ly-35),(lx+10,ly+37),(18,18,18),-1)
        cv2.rectangle(frame,(lx-10,ly-35),(lx+10,ly+37),(55,55,55),1)
        r_on = RED_SIG   if not green_for_peds else (20,20,60)
        g_on = GREEN_SIG if green_for_peds      else (20,50,20)
        cv2.circle(frame,(lx,ly-18),8,r_on,-1)
        cv2.circle(frame,(lx,ly+18),8,g_on,-1)


def draw_hud(frame, t, fps_v, ped_cnt, veh_cnt, green_for_peds, risk):
    # Semi-transparent dark bar
    hud = frame.copy()
    cv2.rectangle(hud,(0,0),(400,130),(8,8,12),-1)
    cv2.addWeighted(hud,0.72,frame,0.28,0,frame)

    cv2.putText(frame,f"FPS: {int(fps_v)}",(12,36),
                cv2.FONT_HERSHEY_SIMPLEX,0.85,SAFE_CLR,2,cv2.LINE_AA)
    cv2.putText(frame,f"Objects: {ped_cnt+veh_cnt}",(12,68),
                cv2.FONT_HERSHEY_SIMPLEX,0.85,ORANGE,2,cv2.LINE_AA)

    zone_lbl  = "OCCUPIED" if ped_cnt > 0 else "CLEAR"
    zone_col  = (0,60,255) if ped_cnt > 0 else (60,220,60)
    zone_text = f"Zone: {zone_lbl}  |  Peds: {ped_cnt}"
    (tw,th),_ = cv2.getTextSize(zone_text,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.rectangle(frame,(10,80),(10+tw+10,80+th+10),(0,0,0),-1)
    cv2.putText(frame,zone_text,(14,98),cv2.FONT_HERSHEY_SIMPLEX,0.6,zone_col,2,cv2.LINE_AA)

    # Top-right info
    sig = "GREEN — PED WALK" if green_for_peds else "RED — VEHICLES GO"
    sig_col = GREEN_SIG if green_for_peds else RED_SIG
    cv2.putText(frame,f"Signal: {sig}",(W-370,36),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,sig_col,1,cv2.LINE_AA)
    cv2.putText(frame,"YOLOv8 | ByteTrack",(W-260,58),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(140,140,160),1,cv2.LINE_AA)
    ts = f"LIVE  T+{t:.1f}s"
    cv2.putText(frame,ts,(W-200,80),cv2.FONT_HERSHEY_SIMPLEX,0.55,(120,255,180),1,cv2.LINE_AA)

    # Risk badge top-centre
    risk_col = HIGH_CLR if risk=="HIGH RISK" else ((0,220,220) if risk=="MEDIUM" else SAFE_CLR)
    badge = f"  RISK: {risk}  "
    (bw,bh),_ = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.85,2)
    bx = (W-bw)//2
    cv2.rectangle(frame,(bx-6,6),(bx+bw+6,6+bh+12),risk_col,-1)
    cv2.putText(frame,badge,(bx,6+bh+4),cv2.FONT_HERSHEY_SIMPLEX,0.85,BLACK,2,cv2.LINE_AA)

    # Location badge bottom-right
    loc = "Shibuya Scramble Crossing  |  Tokyo, Japan"
    cv2.putText(frame,loc,(W-560,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1,cv2.LINE_AA)


# ---- Simulation state -------------------------------------------------------
rng = random.Random(42)
peds = []
vehs = []

# Spawn pedestrians in waves
SPAWN_TIMES = list(range(0, SECS, 6))   # every 6 seconds a wave spawns
wave_spawned = set()

# Initial vehicles (waiting)
for lane in range(2):
    for direction in ['S','N','E','W']:
        vehs.append(Vehicle(lane, direction, rng))

import time as _time

print("[*] Generating Tokyo Shibuya scramble crossing video ...")
prev_t_real = _time.time()

for fi in range(TOTAL):
    t = fi / FPS

    # -- Signal cycle: 10s ped green, 8s vehicle green (repeat)
    cycle = t % 18
    green_for_peds = cycle < 10

    # -- Spawn pedestrian wave at each green phase start
    wave_idx = int(t // 18)
    wave_key = (wave_idx, "ped")
    if green_for_peds and wave_key not in wave_spawned:
        wave_spawned.add(wave_key)
        n_peds = rng.randint(10, 18)
        for _ in range(n_peds):
            peds.append(Pedestrian(rng))

    # -- Spawn vehicle wave at each vehicle-green phase
    veh_key = (wave_idx, "veh")
    if not green_for_peds and veh_key not in wave_spawned:
        wave_spawned.add(veh_key)
        for lane in range(2):
            for direction in ['S','N','E','W']:
                v = Vehicle(lane, direction, rng)
                # stagger start times
                v.x += rng.uniform(-20,20)
                v.y += rng.uniform(-20,20)
                vehs.append(v)

    # -- Step all entities
    peds = [p for p in peds if not p.done]
    vehs = [v for v in vehs if not v.done]
    for p in peds:
        p.step()
    for v in vehs:
        v.step(not green_for_peds)  # vehicles go on vehicle-green (not ped-green)

    # -- Draw frame
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    draw_road(frame)
    draw_crosswalks(frame)

    # Vehicles behind crosswalk (draw first)
    for v in vehs:
        v.draw(frame)

    # Pedestrians on top
    ped_count = 0
    for p in peds:
        if in_any_crosswalk(int(p.x), int(p.y)):
            ped_count += 1
        p.draw(frame)

    # Traffic lights
    draw_traffic_lights(frame, green_for_peds)

    # Distance lines between nearest ped-vehicle pairs (if ped-green AND vehicles still moving)
    risk = "SAFE"
    for v in vehs:
        vx, vy = int(v.x), int(v.y)
        for p in peds:
            px, py = int(p.x), int(p.y)
            if in_any_crosswalk(px, py):
                dist_px = math.sqrt((vx-px)**2+(vy-py)**2)
                dist_m  = dist_px * 0.05
                if dist_m < 12 and not green_for_peds:
                    col = HIGH_CLR if dist_m < 5 else (0,220,220)
                    # Dashed line
                    pts = np.linspace((vx,vy),(px,py),16).astype(int)
                    for i in range(0,len(pts)-1,2):
                        cv2.line(frame,tuple(pts[i]),tuple(pts[i+1]),col,2,cv2.LINE_AA)
                    mid = ((vx+px)//2,(vy+py)//2)
                    cv2.putText(frame,f"{dist_m:.1f}m",mid,
                                cv2.FONT_HERSHEY_SIMPLEX,0.55,WHITE,2,cv2.LINE_AA)
                    if dist_m < 5:
                        risk = "HIGH RISK"
                    elif risk != "HIGH RISK":
                        risk = "MEDIUM"

    # Simulated FPS jitter
    curr_t = _time.time()
    fps_v = 1/(curr_t-prev_t_real) if prev_t_real else FPS
    prev_t_real = curr_t
    # clamp for display
    fps_display = min(FPS+2, max(15, fps_v))

    draw_hud(frame, t, fps_display, ped_count, len(vehs), green_for_peds, risk)

    # Alert banner during vehicle-green with peds still in crosswalk
    if not green_for_peds and ped_count > 0:
        al = frame.copy()
        cv2.rectangle(al,(0,H-55),(W,H),HIGH_CLR,-1)
        cv2.addWeighted(al,0.85,frame,0.15,0,frame)
        msg = "!! PEDESTRIANS STILL IN CROSSWALK -- VEHICLES HOLD !!"
        (tw,_),_ = cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.75,2)
        cv2.putText(frame,msg,((W-tw)//2,H-16),cv2.FONT_HERSHEY_SIMPLEX,0.75,WHITE,2,cv2.LINE_AA)

    writer.write(frame)

writer.release()
print(f"[OK] Video saved -> {OUT_PATH}")
print(f"     Size: {os.path.getsize(OUT_PATH)//1024} KB, Duration: {SECS}s")
