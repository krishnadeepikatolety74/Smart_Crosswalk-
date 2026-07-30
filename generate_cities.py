# -*- coding: utf-8 -*-
"""
Bulk-generate all missing city simulation videos.
Saves to static/videos/<city>.mp4

Cities covered:
  hyderabad.mp4 - Chaotic multi-lane Indian intersection (diverse vehicle types)
  seoul.mp4     - South Korean crosswalk, orderly dense crossing
  bangkok.mp4   - Bangkok Siam Sq, busy tropical intersection
  newyork.mp4   - Times Square style, yellow taxis, dense urban
  london.mp4    - Oxford Street, double-deck buses, organised queues
  singapore.mp4 - Orchard Road, clean orderly wide crosswalk

Run: python generate_cities.py
"""
import cv2
import numpy as np
import math, os, random, time as _time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "static", "videos")
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 1280, 720
FPS  = 30

# ── shared colours ────────────────────────────────────────────────────────────
WHITE   = (255,255,255)
BLACK   = (0,0,0)
YELLOW  = (0,255,255)
ORANGE  = (0,165,255)
MAGENTA = (255,0,255)
SAFE_C  = (60,220,60)
HIGH_C  = (50,50,255)
MED_C   = (0,220,220)

def make_writer(name):
    path = os.path.join(OUT_DIR, name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, FPS, (W, H)), path

def put_pill(frame, text, x, y, fg, bg=(0,0,0), scale=0.6, thick=1):
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick+1)
    cv2.rectangle(frame,(x-4,y-th-6),(x+tw+4,y+4),bg,-1)
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,fg,thick+1,cv2.LINE_AA)

def draw_hud_common(frame, t, fps_v, obj_count, zone_occ, ped_n, risk, loc_label):
    hud = frame.copy()
    cv2.rectangle(hud,(0,0),(420,140),(8,8,12),-1)
    cv2.addWeighted(hud,0.72,frame,0.28,0,frame)
    cv2.putText(frame,f"FPS: {int(fps_v)}",(12,36),cv2.FONT_HERSHEY_SIMPLEX,0.85,SAFE_C,2,cv2.LINE_AA)
    cv2.putText(frame,f"Objects: {obj_count}",(12,68),cv2.FONT_HERSHEY_SIMPLEX,0.85,ORANGE,2,cv2.LINE_AA)
    zone_lbl = "OCCUPIED" if zone_occ else "CLEAR"
    zone_col = (0,60,255) if zone_occ else (60,220,60)
    zone_txt = f"Zone: {zone_lbl}  |  Peds: {ped_n}"
    put_pill(frame, zone_txt, 14, 104, zone_col, scale=0.6, thick=1)
    # top-centre risk badge
    risk_col = HIGH_C if risk=="HIGH RISK" else (MED_C if risk=="MEDIUM" else SAFE_C)
    badge = f"  RISK: {risk}  "
    (bw,bh),_ = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.85,2)
    bx=(W-bw)//2
    cv2.rectangle(frame,(bx-6,6),(bx+bw+6,6+bh+12),risk_col,-1)
    cv2.putText(frame,badge,(bx,6+bh+4),cv2.FONT_HERSHEY_SIMPLEX,0.85,BLACK,2,cv2.LINE_AA)
    # top-right
    cv2.putText(frame,"YOLOv8 | ByteTrack",(W-260,58),cv2.FONT_HERSHEY_SIMPLEX,0.45,(140,140,160),1,cv2.LINE_AA)
    cv2.putText(frame,f"LIVE  T+{t:.1f}s",(W-200,80),cv2.FONT_HERSHEY_SIMPLEX,0.55,(120,255,180),1,cv2.LINE_AA)
    # location badge bottom
    cv2.putText(frame,loc_label,(W-len(loc_label)*9,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1,cv2.LINE_AA)

def draw_cw_zone(frame, x1,y1,x2,y2, alpha=0.15, stripes='v'):
    """Draw pink crosswalk zone with zebra stripes."""
    if stripes == 'v':  # vertical stripes (horizontal road crossing)
        for sx in range(x1+8, x2-8, 32):
            cv2.rectangle(frame,(sx,y1),(sx+18,y2),(195,195,190),-1)
    else:               # horizontal stripes (vertical road crossing)
        for sy in range(y1+8, y2-8, 32):
            cv2.rectangle(frame,(x1,sy),(x2,sy+18),(195,195,190),-1)
    poly = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.int32).reshape(-1,1,2)
    ov = frame.copy()
    cv2.fillPoly(ov,[poly],MAGENTA)
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    cv2.polylines(frame,[poly],True,MAGENTA,2)

def lerp(a,b,t):
    return a + (b-a)*t


# ═══════════════════════════════════════════════════════════════════════════════
# CITY RENDERER BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class CityVideo:
    def __init__(self, filename, secs, theme, loc_label, rng_seed=0):
        self.filename  = filename
        self.secs      = secs
        self.total     = secs * FPS
        self.theme     = theme   # dict of road/sidewalk colours
        self.loc       = loc_label
        self.rng       = random.Random(rng_seed)
        self.peds      = []
        self.vehs      = []
        self._prev_t   = 0.0

    def road_color(self): return self.theme.get('road',(40,40,44))
    def side_color(self): return self.theme.get('side',(68,65,62))
    def accent(self):     return self.theme.get('acc', (137,180,250))

    def draw_road_base(self, frame, cw_x1, cw_x2, cw_y):
        """Single horizontal crosswalk road layout."""
        frame[:] = self.road_color()
        # Sidewalks
        cv2.rectangle(frame,(0,0),(W,cw_y-90), self.side_color(),-1)
        cv2.rectangle(frame,(0,cw_y+90),(W,H),  self.side_color(),-1)
        # Lane dividers
        for lx in range(100, W-100, 80):
            cv2.rectangle(frame,(lx,cw_y-60),(lx+45,cw_y-58),(130,130,135),-1)
            cv2.rectangle(frame,(lx,cw_y+58),(lx+45,cw_y+60),(130,130,135),-1)
        # Kerb lines
        cv2.line(frame,(0,cw_y-90),(W,cw_y-90),(80,78,75),2)
        cv2.line(frame,(0,cw_y+90),(W,cw_y+90),(80,78,75),2)

    def spawn_ped(self, cw_x1,cw_x2,cw_y, n=1, col=None):
        for _ in range(n):
            col_ = col or self.rng.choice([(161,227,166),(137,180,250),(250,179,135),(243,139,168)])
            direction = self.rng.choice(['south','north'])
            if direction == 'south':
                x = self.rng.uniform(cw_x1+10, cw_x2-10)
                self.peds.append({'x':x,'y':float(cw_y-100),'tx':x+self.rng.uniform(-15,15),'ty':float(cw_y+100),
                                  'col':col_,'conf':round(0.87+self.rng.random()*0.11,2),'done':False,'spd':1.3+self.rng.random()*0.4})
            else:
                x = self.rng.uniform(cw_x1+10, cw_x2-10)
                self.peds.append({'x':x,'y':float(cw_y+100),'tx':x+self.rng.uniform(-15,15),'ty':float(cw_y-100),
                                  'col':col_,'conf':round(0.87+self.rng.random()*0.11,2),'done':False,'spd':1.3+self.rng.random()*0.4})

    def spawn_veh(self, cw_y, direction='right', col=None, spd_kmh=None, label='car', width=62, height=110):
        col_ = col or self.rng.choice([(80,130,200),(130,100,200),(200,110,80),(80,200,130)])
        spd  = spd_kmh or self.rng.uniform(25,50)
        lane_offset = self.rng.choice([-40,0,40])
        if direction == 'right':
            self.vehs.append({'x':-80.0,'y':float(cw_y+lane_offset),'tx':float(W+80),'ty':float(cw_y+lane_offset),
                              'col':col_,'conf':round(0.85+self.rng.random()*0.12,2),'done':False,
                              'spd':2.0+self.rng.random()*0.8,'spd_kmh':spd,'label':label,'w':width,'h':height})
        else:
            self.vehs.append({'x':float(W+80),'y':float(cw_y+lane_offset),'tx':-80.0,'ty':float(cw_y+lane_offset),
                              'col':col_,'conf':round(0.85+self.rng.random()*0.12,2),'done':False,
                              'spd':2.0+self.rng.random()*0.8,'spd_kmh':spd,'label':label,'w':width,'h':height})

    def step_entities(self, green_for_peds):
        self.peds = [p for p in self.peds if not p['done']]
        self.vehs = [v for v in self.vehs if not v['done']]
        for p in self.peds:
            dx,dy = p['tx']-p['x'], p['ty']-p['y']
            d = math.sqrt(dx*dx+dy*dy)
            if d<2: p['done']=True; continue
            p['x']+=p['spd']*dx/d; p['y']+=p['spd']*dy/d
        for v in self.vehs:
            if not green_for_peds:  # vehicles move when ped signal is red
                dx,dy = v['tx']-v['x'], v['ty']-v['y']
                d=math.sqrt(dx*dx+dy*dy)
                if d<2: v['done']=True; continue
                v['x']+=v['spd']*dx/d; v['y']+=v['spd']*dy/d

    def draw_entities(self, frame, cw_x1, cw_x2, cw_y):
        # Vehicles
        for v in self.vehs:
            cx,cy = int(v['x']),int(v['y'])
            hw,hh = v['w']//2, v['h']//2
            x1,y1,x2,y2 = cx-hh,cy-hw,cx+hh,cy+hw  # horizontal vehicle
            cv2.rectangle(frame,(x1+3,y1+3),(x2+3,y2+3),(0,0,0),-1)
            cv2.rectangle(frame,(x1,y1),(x2,y2),v['col'],-1)
            cv2.rectangle(frame,(x1,y1),(x2,y2),WHITE,2)
            lbl=f"{v['label']}:{v['conf']:.2f}"
            put_pill(frame,lbl,x1,y1-4,BLACK,v['col'],0.38,1)
            cv2.putText(frame,f"{int(v['spd_kmh'])}km/h",(x1,y1-16),cv2.FONT_HERSHEY_SIMPLEX,0.38,YELLOW,1,cv2.LINE_AA)
        # Pedestrians
        ped_in_cw = 0
        for p in self.peds:
            cx,cy=int(p['x']),int(p['y'])
            x1,y1,x2,y2 = cx-11,cy-24,cx+11,cy+24
            cv2.rectangle(frame,(x1,y1),(x2,y2),p['col'],2)
            cv2.circle(frame,(cx,y1-7),7,p['col'],2)
            lbl=f"person:{p['conf']:.2f}"
            put_pill(frame,lbl,x1,y1-4,BLACK,p['col'],0.36,1)
            if cw_x1<=cx<=cw_x2 and (cw_y-90)<=cy<=(cw_y+90):
                ped_in_cw+=1
        return ped_in_cw

    def compute_risk(self, frame, cw_x1,cw_x2,cw_y, green_for_peds):
        risk="SAFE"
        if green_for_peds:
            return risk
        for v in self.vehs:
            vx,vy=int(v['x']),int(v['y'])
            for p in self.peds:
                px,py=int(p['x']),int(p['y'])
                if cw_x1<=px<=cw_x2 and (cw_y-90)<=py<=(cw_y+90):
                    d=math.sqrt((vx-px)**2+(vy-py)**2)*0.05
                    if d<12:
                        col=HIGH_C if d<5 else MED_C
                        pts=np.linspace((vx,vy),(px,py),14).astype(int)
                        for i in range(0,len(pts)-1,2):
                            cv2.line(frame,tuple(pts[i]),tuple(pts[i+1]),col,2,cv2.LINE_AA)
                        cv2.putText(frame,f"{d:.1f}m",((vx+px)//2,(vy+py)//2),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.55,WHITE,2,cv2.LINE_AA)
                        if d<5: risk="HIGH RISK"
                        elif risk!="HIGH RISK": risk="MEDIUM"
        return risk

    def render(self, extra_fn=None):
        writer, path = make_writer(self.filename)
        print(f"  Rendering {self.filename} ...")
        prev_t = _time.time()
        CW_X1, CW_X2, CW_Y = W//2-180, W//2+180, H//2+30

        for fi in range(self.total):
            t = fi / FPS
            cycle = t % 15
            green_for_peds = cycle < 8

            # Spawn waves
            wave = int(t // 15)
            wave_ped_key = (wave,'p')
            wave_veh_key = (wave,'v')
            if green_for_peds and not hasattr(self, f'_sp_{wave_ped_key}'):
                setattr(self, f'_sp_{wave_ped_key}', True)
                self.spawn_ped(CW_X1, CW_X2, CW_Y, n=self.rng.randint(4,8))
            if not green_for_peds and not hasattr(self, f'_sp_{wave_veh_key}'):
                setattr(self, f'_sp_{wave_veh_key}', True)
                self._spawn_city_vehicles(CW_Y)

            self.step_entities(green_for_peds)

            frame = np.zeros((H,W,3),dtype=np.uint8)
            self.draw_road_base(frame, CW_X1, CW_X2, CW_Y)
            if extra_fn: extra_fn(frame, CW_X1, CW_X2, CW_Y, t)
            draw_cw_zone(frame, CW_X1, CW_X2, CW_Y-90, CW_Y+90, stripes='v')

            ped_in_cw = self.draw_entities(frame, CW_X1, CW_X2, CW_Y)
            risk = self.compute_risk(frame, CW_X1, CW_X2, CW_Y, green_for_peds)

            # Traffic light
            tl_col_r = (0,0,220) if not green_for_peds else (20,20,60)
            tl_col_g = (0,220,60) if green_for_peds    else (20,50,20)
            for tx in [80, W-80]:
                cv2.rectangle(frame,(tx-12,CW_Y-50),(tx+12,CW_Y+12),(18,18,18),-1)
                cv2.circle(frame,(tx,CW_Y-30),9,tl_col_r,-1)
                cv2.circle(frame,(tx,CW_Y   ),9,(20,100,20),-1)
                cv2.circle(frame,(tx,CW_Y+30) if green_for_peds else (tx,CW_Y),9,tl_col_g,-1)

            # Alert banner
            if not green_for_peds and ped_in_cw>0:
                al=frame.copy()
                cv2.rectangle(al,(0,H-55),(W,H),HIGH_C,-1)
                cv2.addWeighted(al,0.85,frame,0.15,0,frame)
                msg="!! PEDESTRIANS IN CROSSWALK -- VEHICLES HOLD !!"
                (tw,_),_=cv2.getTextSize(msg,cv2.FONT_HERSHEY_SIMPLEX,0.75,2)
                cv2.putText(frame,msg,((W-tw)//2,H-16),cv2.FONT_HERSHEY_SIMPLEX,0.75,WHITE,2,cv2.LINE_AA)

            curr_t = _time.time()
            fps_v = min(FPS+2, max(12, 1/(curr_t-prev_t) if prev_t else FPS))
            prev_t = curr_t
            draw_hud_common(frame,t,fps_v,ped_in_cw+len(self.vehs),ped_in_cw>0,ped_in_cw,risk,self.loc)
            writer.write(frame)

        writer.release()
        sz = os.path.getsize(path)//1024
        print(f"  [OK] {self.filename} -> {sz} KB")

    def _spawn_city_vehicles(self, cw_y):
        """Override per city for custom vehicle types."""
        for _ in range(self.rng.randint(3,6)):
            d = self.rng.choice(['right','left'])
            self.spawn_veh(cw_y, direction=d)


# ═══════════════════════════════════════════════════════════════════════════════
# CITY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class HyderabadVideo(CityVideo):
    """Chaotic Indian intersection — autos, bikes, trucks."""
    def _spawn_city_vehicles(self, cw_y):
        VEH_TYPES = [
            ('car',62,110),('car',62,110),('motorcycle',28,55),
            ('motorcycle',28,55),('bus',75,160),('truck',70,145),
        ]
        for _ in range(self.rng.randint(5,9)):
            d = self.rng.choice(['right','left'])
            lbl,w,h = self.rng.choice(VEH_TYPES)
            col = self.rng.choice([(80,130,200),(200,100,80),(130,200,100),(200,200,80),(100,80,200)])
            self.spawn_veh(cw_y,direction=d,col=col,spd_kmh=self.rng.uniform(15,55),label=lbl,width=w,height=h)

class SeoulVideo(CityVideo):
    """Korean crosswalk — orderly, mostly cars and buses."""
    def _spawn_city_vehicles(self, cw_y):
        for _ in range(self.rng.randint(4,7)):
            d=self.rng.choice(['right','left'])
            lbl,w,h=self.rng.choice([('car',62,110),('bus',75,160),('car',62,110)])
            self.spawn_veh(cw_y,direction=d,spd_kmh=self.rng.uniform(30,50),label=lbl,width=w,height=h)

class BangkokVideo(CityVideo):
    """Bangkok — tuk-tuks, motos, buses."""
    def _spawn_city_vehicles(self, cw_y):
        for _ in range(self.rng.randint(5,8)):
            d=self.rng.choice(['right','left'])
            lbl,w,h=self.rng.choice([('car',62,110),('motorcycle',28,55),('motorcycle',28,55),('bus',75,160)])
            col=self.rng.choice([(80,130,200),(0,165,255),(200,80,80),(80,200,200)])
            self.spawn_veh(cw_y,direction=d,col=col,spd_kmh=self.rng.uniform(20,45),label=lbl,width=w,height=h)

class NewYorkVideo(CityVideo):
    """NYC — yellow taxis and buses."""
    def _spawn_city_vehicles(self, cw_y):
        YELLOW_CAB=(0,200,240)
        for _ in range(self.rng.randint(4,7)):
            d=self.rng.choice(['right','left'])
            lbl,w,h=self.rng.choice([('car',62,110),('car',62,110),('bus',75,160),('truck',70,145)])
            col=self.rng.choice([YELLOW_CAB,YELLOW_CAB,(80,80,200),(130,130,130)])
            self.spawn_veh(cw_y,direction=d,col=col,spd_kmh=self.rng.uniform(25,50),label=lbl,width=w,height=h)

class LondonVideo(CityVideo):
    """London — double-decker red buses, orderly."""
    def _spawn_city_vehicles(self, cw_y):
        RED_BUS=(40,40,200)
        for _ in range(self.rng.randint(3,6)):
            d=self.rng.choice(['right','left'])
            lbl,w,h=self.rng.choice([('car',62,110),('bus',80,180),('bus',80,180),('car',62,110)])
            col=self.rng.choice([RED_BUS,RED_BUS,(80,130,200),(140,140,140)])
            self.spawn_veh(cw_y,direction=d,col=col,spd_kmh=self.rng.uniform(20,40),label=lbl,width=w,height=h)

class SingaporeVideo(CityVideo):
    """Singapore — clean, wide, orderly crosswalk."""
    def _spawn_city_vehicles(self, cw_y):
        for _ in range(self.rng.randint(3,5)):
            d=self.rng.choice(['right','left'])
            lbl,w,h=self.rng.choice([('car',62,110),('bus',75,160),('car',62,110)])
            self.spawn_veh(cw_y,direction=d,spd_kmh=self.rng.uniform(30,45),label=lbl,width=w,height=h)


# ── Themes ───────────────────────────────────────────────────────────────────
THEMES = {
    'hyderabad': {'road':(38,36,34), 'side':(72,65,58), 'acc':(0,165,255)},
    'seoul':     {'road':(32,32,38), 'side':(65,64,70), 'acc':(137,180,250)},
    'bangkok':   {'road':(40,38,30), 'side':(72,68,52), 'acc':(0,220,200)},
    'newyork':   {'road':(30,30,34), 'side':(60,60,65), 'acc':(0,230,240)},
    'london':    {'road':(35,33,32), 'side':(65,62,60), 'acc':(40,40,200)},
    'singapore': {'road':(30,32,36), 'side':(62,65,68), 'acc':(100,220,180)},
}

LOCS = {
    'hyderabad': "Gachibowli Flyover  |  Hyderabad, India",
    'seoul':     "Gangnam Station Crosswalk  |  Seoul, South Korea",
    'bangkok':   "Siam Square Intersection  |  Bangkok, Thailand",
    'newyork':   "Times Square  |  New York City, USA",
    'london':    "Oxford Street  |  London, United Kingdom",
    'singapore': "Orchard Road Crossing  |  Singapore",
}

CITIES = [
    HyderabadVideo('hyderabad.mp4', 25, THEMES['hyderabad'], LOCS['hyderabad'], 10),
    SeoulVideo    ('seoul.mp4',     25, THEMES['seoul'],     LOCS['seoul'],     20),
    BangkokVideo  ('bangkok.mp4',   25, THEMES['bangkok'],   LOCS['bangkok'],   30),
    NewYorkVideo  ('newyork.mp4',   25, THEMES['newyork'],   LOCS['newyork'],   40),
    LondonVideo   ('london.mp4',    25, THEMES['london'],    LOCS['london'],    50),
    SingaporeVideo('singapore.mp4', 25, THEMES['singapore'], LOCS['singapore'], 60),
]

print("[*] Generating all city videos ...")
for city in CITIES:
    city.render()

print("\n[DONE] All city videos generated in static/videos/")
