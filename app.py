import math, sys, time, urllib.request
from collections import deque
from pathlib import Path
import cv2, mediapipe as mp, numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from database import Database
from recorder import RollingVideoBuffer
from risk_engine import RiskEngine
from settings import *

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / MODEL_DIR / "face_landmarker.task"
DB_PATH = APP_DIR / DATABASE_NAME
INCIDENT_PATH = APP_DIR / INCIDENTS_DIR
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# ---------------------------------------------------------------------------
# GPS & TELEPHONY DISPATCH SIMULATION HELPERS
# ---------------------------------------------------------------------------

def get_current_gps_location():
    """
    Simulates retrieving vehicle coordinates from a hardware GPS module.
    Returns latitude and longitude data.
    """
    return {
        "latitude": 25.3223,
        "longitude": 55.5136,
        "timestamp": time.time()
    }

def trigger_emergency_phone_dispatch(driver_name, risk_score, gps_info, video_path):
    """
    Simulates sending the GPS location and video clip payload to the emergency 
    call dispatch and telephony system.
    """
    print("\n" + "=" * 60)
    print(" [EMERGENCY SYSTEM ACTIVATED]")
    print(f" Driver Name : {driver_name}")
    print(f" Risk Score  : {risk_score:.0f}%")
    print(f" GPS Location: Lat {gps_info['latitude']}, Lng {gps_info['longitude']}")
    print(f" Video Clip  : {video_path}")
    print(" Dispatching payload to Telephony Call System & Emergency Services...")
    print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# EYVORA CORE UTILITIES
# ---------------------------------------------------------------------------

def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe face model (one-time setup)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def blendshape_dict(result):
    if not result.face_blendshapes: return {}
    return {c.category_name: float(c.score) for c in result.face_blendshapes[0]}

def rotation_to_euler(matrix):
    r = matrix[:3,:3].astype(np.float64)
    sy = math.sqrt(r[0,0]**2 + r[1,0]**2)
    if sy >= 1e-6:
        pitch = math.atan2(r[2,1], r[2,2])
        yaw = math.atan2(-r[2,0], sy)
        roll = math.atan2(r[1,0], r[0,0])
    else:
        pitch = math.atan2(-r[1,2], r[1,1])
        yaw = math.atan2(-r[2,0], sy)
        roll = 0.0
    return tuple(math.degrees(x) for x in (pitch,yaw,roll))

def get_head_angles(result):
    if not result.facial_transformation_matrixes: return None
    m = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64).reshape(4,4)
    return rotation_to_euler(m)

def open_camera(index):
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    elif sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cap

def draw_face(frame, landmarks):
    h,w = frame.shape[:2]
    xs, ys = [p.x for p in landmarks], [p.y for p in landmarks]
    cv2.rectangle(frame, (int(min(xs)*w),int(min(ys)*h)), (int(max(xs)*w),int(max(ys)*h)), (70,220,120), 2)

class DriverMonitor:
    def __init__(self, db, session_id):
        self.db, self.session_id = db, session_id
        self.blinks = self.yawns = 0
        self.eye_closed_since = self.yawn_since = None
        self.yawn_counted = False
        self.neutral = None
        self.no_face_since = self.head_away_since = None
        self.drowsy_logged = self.head_logged = self.no_face_logged = False
        self.recent_yawns = deque(maxlen=20)

    def centre_head(self, result):
        a = get_head_angles(result)
        if a:
            self.neutral = a
            self.db.log_event(self.session_id, "head_recentered")

    def _recent_yawns(self, now):
        while self.recent_yawns and now - self.recent_yawns[0] > 60:
            self.recent_yawns.popleft()
        return len(self.recent_yawns)

    def process(self, result, now):
        if not result.face_landmarks:
            if self.no_face_since is None: self.no_face_since = now
            if now - self.no_face_since >= NO_FACE_ALERT_SECONDS and not self.no_face_logged:
                self.db.log_event(self.session_id, "no_face", severity="warning")
                self.no_face_logged = True
            self.eye_closed_since = self.yawn_since = None
            return dict(face=False, eyes_closed=False, closed_seconds=0.0, drowsy=False,
                        yawning=False, head_direction="--", head_away_alert=False,
                        blinks=self.blinks, yawns=self.yawns, yawns_recent=self._recent_yawns(now))
        self.no_face_since, self.no_face_logged = None, False

        s = blendshape_dict(result)
        eye = (s.get("eyeBlinkLeft",0)+s.get("eyeBlinkRight",0))/2
        closed = eye >= EYE_CLOSED_THRESHOLD
        closed_seconds = 0.0
        if closed:
            if self.eye_closed_since is None: self.eye_closed_since = now
            closed_seconds = now - self.eye_closed_since
        else:
            if self.eye_closed_since is not None:
                dur = now - self.eye_closed_since
                if BLINK_MIN_SECONDS <= dur <= BLINK_MAX_SECONDS: self.blinks += 1
            self.eye_closed_since = None

        drowsy = closed and closed_seconds >= DROWSY_SECONDS
        if drowsy and not self.drowsy_logged:
            self.db.log_event(self.session_id, "drowsiness", f"{closed_seconds:.2f}s", "critical")
            self.drowsy_logged = True
        elif not drowsy:
            self.drowsy_logged = False

        jaw = s.get("jawOpen",0)
        mouth_open = jaw >= YAWN_JAW_THRESHOLD
        if mouth_open:
            if self.yawn_since is None: self.yawn_since = now
            yawning = now - self.yawn_since >= YAWN_MIN_SECONDS
            if yawning and not self.yawn_counted:
                self.yawns += 1
                self.recent_yawns.append(now)
                self.yawn_counted = True
                self.db.log_event(self.session_id, "yawn", f"#{self.yawns}", "warning")
        else:
            yawning = False
            self.yawn_since = None
            self.yawn_counted = False

        pitch=yaw=roll=0.0
        angles = get_head_angles(result)
        if angles:
            if self.neutral is None: self.neutral = angles
            pitch = PITCH_SIGN*(angles[0]-self.neutral[0])
            yaw = YAW_SIGN*(angles[1]-self.neutral[1])
            roll = angles[2]-self.neutral[2]

        if yaw > HEAD_YAW_THRESHOLD_DEG: direction="RIGHT"
        elif yaw < -HEAD_YAW_THRESHOLD_DEG: direction="LEFT"
        elif pitch > HEAD_PITCH_THRESHOLD_DEG: direction="DOWN"
        elif pitch < -HEAD_PITCH_THRESHOLD_DEG: direction="UP"
        else: direction="FORWARD"

        if direction != "FORWARD":
            if self.head_away_since is None: self.head_away_since=now
            head_alert = now-self.head_away_since >= HEAD_AWAY_ALERT_SECONDS
            if head_alert and not self.head_logged:
                self.db.log_event(self.session_id, "head_away", direction, "warning")
                self.head_logged=True
        else:
            self.head_away_since=None
            self.head_logged=False
            head_alert=False

        return dict(face=True, eyes_closed=closed, closed_seconds=closed_seconds, drowsy=drowsy,
                    yawning=yawning, head_direction=direction, head_away_alert=head_alert,
                    blinks=self.blinks, yawns=self.yawns, yawns_recent=self._recent_yawns(now))

class EmergencyController:
    def __init__(self, db, session_id):
        self.db, self.session_id = db, session_id
        self.critical_since=None
        self.active=False
        self.started_at=None
        self.cooldown_until=0.0

    def update(self, now, risk):
        if self.active:
            return "COUNTDOWN", max(0.0, EMERGENCY_COUNTDOWN_SECONDS-(now-self.started_at))
        if now < self.cooldown_until:
            return "COOLDOWN",0
        if risk >= CRITICAL_RISK:
            if self.critical_since is None: self.critical_since=now
            if now-self.critical_since >= CRITICAL_HOLD_SECONDS:
                self.active=True
                self.started_at=now
                self.db.log_event(self.session_id, "emergency_countdown_started", severity="critical", risk_score=risk)
                return "COUNTDOWN", EMERGENCY_COUNTDOWN_SECONDS
        else:
            self.critical_since=None
        return "IDLE",0

    def cancel(self, now, risk):
        if not self.active: return False
        self.active=False
        self.critical_since=None
        self.cooldown_until=now+INCIDENT_COOLDOWN_SECONDS
        self.db.log_incident(self.session_id, "driver_alert", risk, "DRIVER_RESPONDED")
        return True

    def expired(self, now):
        return self.active and now-self.started_at >= EMERGENCY_COUNTDOWN_SECONDS

    def resolve(self, now):
        self.active=False
        self.critical_since=None
        self.cooldown_until=now+INCIDENT_COOLDOWN_SECONDS

def draw_dashboard(frame, driver_name, status, risk, state, remaining, fps):
    h,w = frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w,65),(15,20,30),-1)
    cv2.putText(frame,"EYVORA V2",(22,42),cv2.FONT_HERSHEY_SIMPLEX,0.95,(255,255,255),2)
    cv2.putText(frame,f"Driver: {driver_name}",(200,40),cv2.FONT_HERSHEY_SIMPLEX,0.55,(190,200,210),1)

    x,y=18,85
    cv2.rectangle(frame,(x,y),(x+350,y+330),(20,25,35),-1)
    lines=[
        ("DRIVER","DETECTED" if status["face"] else "NO FACE"),
        ("EYES","CLOSED" if status["eyes_closed"] else "OPEN"),
        ("BLINKS",str(status["blinks"])),
        ("YAWNING","YES" if status["yawning"] else "NO"),
        ("YAWNS/60s",str(status["yawns_recent"])),
        ("HEAD",status["head_direction"]),
    ]
    yy=y+38
    for a,b in lines:
        cv2.putText(frame,a,(x+15,yy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(165,178,190),1)
        cv2.putText(frame,b,(x+155,yy),cv2.FONT_HERSHEY_SIMPLEX,0.55,(240,240,240),2)
        yy+=44

    cv2.rectangle(frame,(18,435),(368,565),(20,25,35),-1)
    cv2.putText(frame,"DRIVER RISK",(35,468),cv2.FONT_HERSHEY_SIMPLEX,0.55,(185,195,205),1)
    cv2.putText(frame,f"{risk.score:.0f}%  {risk.level}",(35,505),cv2.FONT_HERSHEY_SIMPLEX,0.78,(255,255,255),2)
    cv2.rectangle(frame,(35,525),(345,548),(70,70,80),-1)
    colors={"SAFE":(80,200,100),"CAUTION":(0,210,255),"HIGH":(0,130,255),"CRITICAL":(60,60,240)}
    fill=int(310*risk.score/100)
    if fill: cv2.rectangle(frame,(35,525),(35+fill,548),colors[risk.level],-1)

    cv2.putText(frame,"C centre | I/SPACE I'm OK | R reset risk | Q quit",
                (18,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.47,(205,210,220),1)

    if state=="COUNTDOWN":
        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(w,h),(10,10,130),-1)
        cv2.addWeighted(ov,0.52,frame,0.48,0,frame)
        cv2.putText(frame,"POSSIBLE DRIVER EMERGENCY",(w//2-285,h//2-80),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),3)
        cv2.putText(frame,"ARE YOU OK?",(w//2-145,h//2-20),
                    cv2.FONT_HERSHEY_SIMPLEX,1.15,(255,255,255),3)
        cv2.putText(frame,str(int(math.ceil(remaining))),(w//2-35,h//2+55),
                    cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255),4)
        cv2.rectangle(frame,(w//2-150,h//2+90),(w//2+150,h//2+145),(65,180,80),-1)
        cv2.putText(frame,"PRESS I / SPACE - I'M OK",(w//2-132,h//2+126),
                    cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,255,255),2)

def main():
    ensure_model()
    driver_name=input("Driver name (Enter for Default Driver): ").strip() or "Default Driver"

    db=Database(DB_PATH)
    driver_id=db.get_or_create_driver(driver_name)
    session_id=db.start_session(driver_id)
    monitor=DriverMonitor(db,session_id)
    risk_engine=RiskEngine()
    emergency=EmergencyController(db,session_id)
    video_buffer=RollingVideoBuffer(ROLLING_BUFFER_SECONDS,ROLLING_BUFFER_MAX_FPS)

    opts=vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    cap=open_camera(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS camera permission.")

    start=time.perf_counter()
    last=time.perf_counter()
    last_ms=-1
    fps=0.0
    last_level="SAFE"

    try:
        with vision.FaceLandmarker.create_from_options(opts) as landmarker:
            while True:
                ok,frame=cap.read()
                if not ok: break
                frame=cv2.flip(frame,1)
                raw=frame.copy()
                now=time.perf_counter()
                dt=max(0.001,now-last); last=now
                inst=1/dt; fps=inst if fps==0 else 0.9*fps+0.1*inst

                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                ms=int((now-start)*1000)
                if ms<=last_ms: ms=last_ms+1
                last_ms=ms

                result=landmarker.detect_for_video(mp_img,ms)
                status=monitor.process(result,now)
                if result.face_landmarks: draw_face(frame,result.face_landmarks[0])

                risk=risk_engine.update(status,dt)
                if risk.level != last_level:
                    db.log_event(session_id,"risk_level_changed",f"{last_level}->{risk.level}",
                                 "warning" if risk.level!="SAFE" else "info",risk.score)
                    last_level=risk.level

                state,remaining=emergency.update(now,risk.score)

                # TRIGGERED ON UNRESPONSIVE DRIVER EMERGENCY EXPIRY
                if emergency.expired(now):
                    clip=video_buffer.save(INCIDENT_PATH,fps)
                    
                    # 1. Fetch simulated vehicle GPS coordinates
                    gps_info = get_current_gps_location()
                    
                    # 2. Trigger simulated phone call & emergency alert payload
                    trigger_emergency_phone_dispatch(driver_name, risk.score, gps_info, clip)
                    
                    # 3. Log event with GPS metadata into local SQLite database
                    db.log_incident(
                        session_id,
                        "unresponsive_driver",
                        risk.score,
                        "NO_RESPONSE",
                        clip,
                        f"Emergency call dispatched. GPS: Lat {gps_info['latitude']}, Lng {gps_info['longitude']}"
                    )
                    emergency.resolve(now)
                    risk_engine.reset()

                video_buffer.add(raw,now)
                state,remaining=emergency.update(now,risk.score)
                draw_dashboard(frame,driver_name,status,risk,state,remaining,fps)
                cv2.imshow("EYVORA V2",frame)
                key=cv2.waitKey(1)&0xFF

                if key in (ord("q"),ord("Q"),27): break
                if key in (ord("c"),ord("C")) and result.face_landmarks: monitor.centre_head(result)
                if key in (ord("i"),ord("I"),32):
                    if emergency.cancel(now,risk.score): risk_engine.reset()
                if key in (ord("r"),ord("R")): risk_engine.reset()
    finally:
        cap.release(); cv2.destroyAllWindows()
        db.end_session(session_id,monitor.blinks,monitor.yawns,risk_engine.max_score)
        db.close()

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print("\nEYVORA ERROR\n------------\n",e)
