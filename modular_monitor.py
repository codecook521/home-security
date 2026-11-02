#!/usr/bin/env python3
"""
Vikirna Unified Modular Security Monitor & PTZ Controller
- Native support for templates/index.html (restoring your professional UI)
- Fully decoupled VideoWriter: Records clean, pristine frames WITHOUT bounding boxes
- Live preview displays YOLO and DeepFace recognition grids
- Modular camera pipeline (RTSP, MJPEG, ONVIF PTZ) driven by cameras.json
- Preserves /api/recordings/dates, /api/recordings/list, and /recordings/video routes
"""

import os
import re
import csv
import glob
import json
import time
import queue
import urllib.request
import threading
import subprocess
import datetime
from io import BytesIO
from PIL import Image

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, jsonify, send_file
from ultralytics import YOLO
from deepface import DeepFace

try:
    from onvif import ONVIFCamera
except ImportError:
    ONVIFCamera = None

# ================= Paths & Global Setup =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "cameras.json")
EVENTS_DIR = os.path.join(BASE_DIR, "events")
CACHE_DIR = os.path.join(EVENTS_DIR, ".cache")
FACES_DIR = os.path.join(BASE_DIR, "faces")
DETECTIONS_LOG = os.path.join(BASE_DIR, "detections.csv")
LOGO_URL = "https://www.vikirna.com/logo.png"

TARGET_CLASSES = {"person", "cat", "dog", "car", "motorcycle"}
YOLO_CONFIDENCE = 0.45
FACE_MODEL = "Facenet512"
FACE_DETECTOR = "opencv"
FACE_THRESHOLD = 0.32
FACE_MARGIN = 0.045
FACE_EXPIRE = 3.5

HOLD_RECORD_SECONDS = 12.0
MOTION_THRESHOLD = 0.8

os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

app = Flask(__name__)

def log(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}][{tag}] {msg}", flush=True)

# ================= Shared Branding Logo =================

def load_branding_logo():
    try:
        req = urllib.request.Request(LOGO_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as res:
            pil_logo = Image.open(BytesIO(res.read())).convert("RGBA")
            aspect = pil_logo.height / pil_logo.width
            new_w = 30
            new_h = int(new_w * aspect)
            pil_logo = pil_logo.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return cv2.cvtColor(np.array(pil_logo), cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        log("Logo", f"Warning loading online logo: {e}")
    return None

GLOBAL_LOGO = load_branding_logo()

def apply_professional_hud(frame, cam_name, is_rec=False, motion_detected=False, subtitle=""):
    """Applies the elegant bottom translucent HUD bar and logo, identical to your original UI."""
    h, w = frame.shape[:2]

    # Logo watermark at top-right
    if GLOBAL_LOGO is not None:
        lh, lw = GLOBAL_LOGO.shape[:2]
        y_offset, x_offset = 12, w - lw - 14
        if y_offset + lh < h and x_offset + lw < w:
            alpha_s = GLOBAL_LOGO[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                frame[y_offset:y_offset+lh, x_offset:x_offset+lw, c] = (
                    alpha_s * GLOBAL_LOGO[:, :, c] + alpha_l * frame[y_offset:y_offset+lh, x_offset:x_offset+lw, c]
                )

    # Sleek bottom bar overlay
    bar_h = 32
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S IST")
    left_hud = f"VIKIRNA MEDIATEK  |  {cam_name.upper()}  |  {ts}"

    if is_rec:
        right_hud = "[REC ACTIVE]  |  LIVE"
        hud_color = (0, 0, 255)
    elif motion_detected:
        right_hud = "[MOTION DETECTED]  |  LIVE"
        hud_color = (0, 140, 255)
    else:
        right_hud = f"{subtitle or f'{w}x{h}'}  |  LIVE"
        hud_color = (0, 215, 255)

    cv2.putText(frame, left_hud, (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
    t_w = cv2.getTextSize(right_hud, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    cv2.putText(frame, right_hud, (w - t_w - 14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, hud_color, 1, cv2.LINE_AA)

    return frame

# ================= Shared Face Database & Sightings Logger =================

class FaceManager:
    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        self.known_embeddings = {}
        self.load()

    def load(self):
        self.known_embeddings = {}
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            files.extend(glob.glob(os.path.join(self.faces_dir, ext)))

        for fp in files:
            raw = os.path.splitext(os.path.basename(fp))[0]
            prefix = raw.split('_')[0].split('-')[0]
            clean = ''.join([c for c in prefix if not c.isdigit()]).strip().title()
            if not clean:
                clean = raw.title()
            try:
                data = DeepFace.represent(img_path=fp, model_name=FACE_MODEL,
                                         detector_backend=FACE_DETECTOR, enforce_detection=False)
                if data:
                    emb = np.array(data[0]["embedding"])
                    if clean not in self.known_embeddings:
                        self.known_embeddings[clean] = []
                    self.known_embeddings[clean].append(emb)
                    log("FaceDB", f"Indexed '{os.path.basename(fp)}' as: {clean}")
            except Exception as e:
                log("FaceDB", f"Failed {fp}: {e}")

FACE_DB = FaceManager(FACES_DIR)

class DetectionsLogger:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.cooldown = {}
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["Timestamp", "Camera", "Category", "Label", "Confidence", "Details"])

    def record(self, cam, cat, label, conf=1.0, details=""):
        now = time.monotonic()
        key = f"{cam}_{label}"
        if now - self.cooldown.get(key, 0) < 3.0:
            return
        self.cooldown[key] = now
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([stamp, cam, cat, label, f"{conf:.2f}", details])

LOGGER = DetectionsLogger(DETECTIONS_LOG)

# ================= Modular Camera Pipeline =================

class CameraWorker(threading.Thread):
    def __init__(self, config, yolo_model):
        super().__init__(daemon=True)
        self.cfg = config
        self.cam_id = config["id"]
        self.name = config["name"]
        self.url = config["url"]
        self.cam_type = config.get("type", "mjpeg")
        self.fps_limit = float(config.get("fps_limit", 10.0))
        self.frame_interval = 1.0 / self.fps_limit
        self.yolo = yolo_model

        self.running = threading.Event()
        self.running.set()

        # Stream & Internal Proxy
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()
        self.frame_cond = threading.Condition(self.frame_lock)

        # Active tracks & Motion states
        self.active_tracks = {}
        self.track_lock = threading.Lock()
        self.motion_detected = False
        self.is_recording = False
        self.video_writer = None
        self.current_video_file = None
        self.last_detection_time = 0.0
        self.last_write_time = 0.0
        self.prev_gray = None

        # PTZ Client
        self.ptz = None
        self.ptz_service = None
        self.ptz_profile = None
        if self.cfg.get("has_ptz") and ONVIFCamera and "onvif" in self.cfg:
            self._init_ptz()

    def _init_ptz(self):
        try:
            o = self.cfg["onvif"]
            self.ptz = ONVIFCamera(o["ip"], o["port"], o["user"], o["pass"])
            media = self.ptz.create_media_service()
            self.ptz_service = self.ptz.create_ptz_service()
            self.ptz_profile = media.GetProfiles()[0]
            log(self.cam_id, "ONVIF PTZ ready.")
        except Exception as e:
            log(self.cam_id, f"ONVIF Init failed: {e}")

    def ptz_move(self, pan, tilt, zoom=0.0):
        if not self.ptz_service or not self.ptz_profile:
            return False
        try:
            req = {
                'ProfileToken': self.ptz_profile.token,
                'Velocity': {'PanTilt': {'x': float(pan), 'y': float(tilt)}, 'Zoom': {'x': float(zoom)}}
            }
            self.ptz_service.ContinuousMove(req)
            return True
        except Exception as e:
            log(self.cam_id, f"PTZ Move error: {e}")
            return False

    def ptz_stop(self):
        if not self.ptz_service or not self.ptz_profile:
            return False
        try:
            self.ptz_service.Stop({'ProfileToken': self.ptz_profile.token, 'PanTilt': True, 'Zoom': True})
            return True
        except Exception:
            return False

    def detect_motion(self, gray):
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        if self.prev_gray is None or self.prev_gray.shape != small.shape:
            self.prev_gray = small
            return False
        diff = cv2.absdiff(self.prev_gray, small)
        self.prev_gray = small
        _, thresh = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
        ratio = (np.count_nonzero(thresh) / thresh.size) * 100.0
        return ratio > MOTION_THRESHOLD

    def match_face(self, crop):
        try:
            res = DeepFace.represent(img_path=crop, model_name=FACE_MODEL,
                                     detector_backend=FACE_DETECTOR, enforce_detection=False)
            if not res:
                return "Unknown", 1.0
            face_emb = np.array(res[0]["embedding"])
            norm_face = np.linalg.norm(face_emb)

            scores = []
            for name, emb_list in FACE_DB.known_embeddings.items():
                best_d = 1.0
                for ke in emb_list:
                    nk = np.linalg.norm(ke)
                    if norm_face == 0 or nk == 0:
                        continue
                    d = 1.0 - (np.dot(face_emb, ke) / (norm_face * nk))
                    if d < best_d:
                        best_d = d
                scores.append((best_d, name))

            if not scores:
                return "Unknown", 1.0

            scores.sort(key=lambda s: s[0])
            best_dist, best_name = scores[0]

            margin_ok = True
            if len(scores) > 1 and (scores[1][0] - best_dist) < FACE_MARGIN:
                margin_ok = False

            if best_dist <= FACE_THRESHOLD and margin_ok:
                return best_name, best_dist
            return "Unknown", best_dist
        except Exception:
            return "Unknown", 1.0

    def start_recording(self, frame):
        if self.is_recording:
            return
        h, w = frame.shape[:2]
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "doorcam" if "door" in self.cam_id.lower() else "event"
        self.current_video_file = os.path.join(EVENTS_DIR, f"{prefix}_{stamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.current_video_file, fourcc, self.fps_limit, (w, h))
        self.is_recording = True
        self.last_detection_time = time.monotonic()
        self.last_write_time = self.last_detection_time
        log(self.cam_id, f"Started clean clip: {os.path.basename(self.current_video_file)}")

    def stop_recording(self):
        if not self.is_recording:
            return
        fp = self.current_video_file
        w = self.video_writer
        self.video_writer = None
        self.is_recording = False
        self.current_video_file = None
        if w:
            w.release()
            log(self.cam_id, f"Finalized clip: {os.path.basename(fp)}")
            def _trans(src):
                tmp = src + ".h264.mp4"
                cmd = ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-preset", "ultrafast",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                    os.replace(tmp, src)
            threading.Thread(target=_trans, args=(fp,), daemon=True).start()

    def process_pipeline(self, raw_frame):
        now = time.monotonic()
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        has_motion = self.detect_motion(gray)
        self.motion_detected = has_motion

        run_yolo = self.is_recording or has_motion
        detected = False
        detections = []

        if run_yolo:
            try:
                results = self.yolo(raw_frame, conf=YOLO_CONFIDENCE, verbose=False)
                for box in results[0].boxes:
                    cname = self.yolo.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cname in TARGET_CLASSES:
                        detected = True
                        LOGGER.record(self.name, "OBJECT", cname, conf)
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        detections.append((cname, conf, xyxy))

                        if cname == "person":
                            x1, y1 = max(0, xyxy[0]), max(0, xyxy[1])
                            x2, y2 = min(raw_frame.shape[1], xyxy[2]), min(raw_frame.shape[0], xyxy[3])
                            if (x2 - x1) > 40 and (y2 - y1) > 40:
                                crop = raw_frame[y1:y2, x1:x2]
                                name, dist = self.match_face(crop)
                                with self.track_lock:
                                    self.active_tracks[f"{x1}_{y1}"] = {
                                        "name": name,
                                        "box": (x1, y1, x2, y2),
                                        "expire": now + FACE_EXPIRE,
                                        "color": (0, 255, 0) if name != "Unknown" else (0, 140, 255)
                                    }
                                LOGGER.record(self.name, "FACE", name, 1.0 - dist)
            except Exception as e:
                log(self.cam_id, f"YOLO error: {e}")

        # Update event recording triggers
        if detected:
            self.last_detection_time = now
            if not self.is_recording:
                self.start_recording(raw_frame)

        # ===================================================================
        # 1. CLEAN RECORDING FRAME: (Pristine video with NO bounding boxes)
        # ===================================================================
        if self.is_recording and self.video_writer is not None:
            clean_disk_frame = raw_frame.copy()
            clean_disk_frame = apply_professional_hud(clean_disk_frame, self.name, is_rec=True, motion_detected=True)
            if now - self.last_write_time >= (self.frame_interval * 0.85):
                self.video_writer.write(clean_disk_frame)
                self.last_write_time = now
            if now - self.last_detection_time >= HOLD_RECORD_SECONDS:
                self.stop_recording()

        # ===================================================================
        # 2. LIVE PREVIEW FRAME: (Overlay YOLO boxes & face tags for user display)
        # ===================================================================
        preview_frame = raw_frame.copy()

        # Draw YOLO target boxes
        for cname, conf, (bx1, by1, bx2, by2) in detections:
            color = (0, 255, 120) if cname == "person" else (255, 180, 0)
            cv2.rectangle(preview_frame, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(preview_frame, f"{cname} {int(conf*100)}%", (bx1, max(18, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw DeepFace tags
        with self.track_lock:
            for k in list(self.active_tracks.keys()):
                t = self.active_tracks[k]
                if now < t["expire"]:
                    fx1, fy1, fx2, fy2 = t["box"]
                    cv2.rectangle(preview_frame, (fx1, fy1), (fx2, fy2), t["color"], 2)
                    cv2.putText(preview_frame, t["name"], (fx1 + 4, max(20, fy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, t["color"], 2, cv2.LINE_AA)
                else:
                    del self.active_tracks[k]

        preview_frame = apply_professional_hud(preview_frame, self.name, is_rec=self.is_recording, motion_detected=self.motion_detected)

        ok, encoded = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self.frame_cond:
                self.latest_jpeg = encoded.tobytes()
                self.frame_cond.notify_all()

    def run_mjpeg(self):
        while self.running.is_set():
            try:
                log(self.cam_id, f"Opening MJPEG: {self.url}")
                req = urllib.request.Request(self.url, headers={"User-Agent": "VikirnaMonitor"})
                stream = urllib.request.urlopen(req, timeout=5.0)
                buf = bytearray()
                log(self.cam_id, "MJPEG connected.")
                last_grab = time.monotonic()

                while self.running.is_set():
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)

                    while True:
                        start = buf.find(b"\xff\xd8")
                        if start < 0:
                            if len(buf) > 1:
                                del buf[:-1]
                            break
                        end = buf.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            if start:
                                del buf[:start]
                            break

                        jpg = bytes(buf[start:end + 2])
                        del buf[:end + 2]

                        now = time.monotonic()
                        if now - last_grab >= self.frame_interval:
                            last_grab = now
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                self.process_pipeline(frame)

                    if len(buf) > 2 * 1024 * 1024:
                        buf.clear()
            except Exception as e:
                log(self.cam_id, f"MJPEG stream error: {e}")
                if self.is_recording:
                    self.stop_recording()
                time.sleep(3.0)

    def run_rtsp(self):
        while self.running.is_set():
            log(self.cam_id, f"Opening RTSP: {self.url}")
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(3.0)
                continue
            log(self.cam_id, "RTSP connected.")
            last_grab = time.monotonic()

            while self.running.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                now = time.monotonic()
                if now - last_grab >= self.frame_interval:
                    last_grab = now
                    self.process_pipeline(frame)
            cap.release()
            if self.is_recording:
                self.stop_recording()
            time.sleep(3.0)

    def run(self):
        if self.cam_type == "rtsp":
            self.run_rtsp()
        else:
            self.run_mjpeg()

    def get_latest_jpeg(self, prev=None, timeout=1.5):
        with self.frame_cond:
            if self.latest_jpeg is None or self.latest_jpeg is prev:
                self.frame_cond.wait(timeout=timeout)
            return self.latest_jpeg

    def stop(self):
        self.running.clear()
        if self.is_recording:
            self.stop_recording()

# ================= Camera Registry & Manager =================

CAMERA_WORKERS = {}
SHARED_YOLO = YOLO("yolov8n.pt")

def load_cameras():
    global CAMERA_WORKERS
    cfg_to_use = CONFIG_FILE if os.path.exists(CONFIG_FILE) else os.path.join(BASE_DIR, "cameras.example.json")
    if not os.path.exists(cfg_to_use):
        log("Manager", f"No camera configuration file found.")
        return
    with open(cfg_to_use, "r") as f:
        configs = json.load(f)

    active_ids = {c["id"] for c in configs}
    for cid in list(CAMERA_WORKERS.keys()):
        if cid not in active_ids:
            log("Manager", f"Stopping removed camera {cid}")
            CAMERA_WORKERS[cid].stop()
            del CAMERA_WORKERS[cid]

    for c in configs:
        cid = c["id"]
        if cid not in CAMERA_WORKERS:
            log("Manager", f"Starting new camera {cid} ({c['name']})")
            worker = CameraWorker(c, SHARED_YOLO)
            worker.start()
            CAMERA_WORKERS[cid] = worker

# ================= File Metadata Helper =================

def parse_file_meta(file_path):
    fname = os.path.basename(file_path)
    is_doorcam = fname.lower().startswith("doorcam_") or fname.lower().endswith(".avi") or "door" in fname.lower()

    m1 = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", fname)
    if m1:
        y, m, d, hh, mm, ss = m1.groups()
        return f"{y}-{m}-{d}", f"{hh}:{mm}:{ss}", "doorcam" if is_doorcam else "cpplus"

    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})[_\s](\d{2})[-:](\d{2})[-:](\d{2})", fname)
    if m2:
        y, m, d, hh, mm, ss = m2.groups()
        return f"{y}-{m}-{d}", f"{hh}:{mm}:{ss}", "doorcam" if is_doorcam else "cpplus"

    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    return mtime.strftime("%Y-%m-%d"), mtime.strftime("%H:%M:%S"), "doorcam" if is_doorcam else "cpplus"

# ================= HTTP & API Routes (Matched to index.html) =================

@app.route('/')
def index():
    cam_ip = ""
    doorcam_ip = ""
    
    # Extract IP dynamically from active workers
    ptz_w = CAMERA_WORKERS.get("ptz_gate") or CAMERA_WORKERS.get("cpplus")
    if ptz_w:
        cam_ip = ptz_w.cfg.get("onvif", {}).get("ip") or ptz_w.url.split("@")[-1].split(":")[0]
        
    door_w = CAMERA_WORKERS.get("doorcam")
    if door_w:
        # Extracts host from http://<cam-ip>:<port>/
        door_url = door_w.url.replace("http://", "").replace("https://", "")
        doorcam_ip = door_url.split(":")[0].split("/")[0]

    return render_template('index.html', cam_ip=cam_ip, doorcam_ip=doorcam_ip)
@app.route('/video_feed')
def video_feed():
    """CP Plus or Main PTZ gate camera feed for index.html."""
    worker = CAMERA_WORKERS.get("ptz_gate") or CAMERA_WORKERS.get("cpplus")
    if not worker and CAMERA_WORKERS:
        worker = list(CAMERA_WORKERS.values())[0]

    def gen():
        prev = None
        while True:
            jpeg = worker.get_latest_jpeg(prev) if worker else None
            if jpeg is None:
                time.sleep(0.04)
                continue
            prev = jpeg
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/doorcam_feed')
def doorcam_feed():
    """Doorcam proxy stream for index.html."""
    worker = CAMERA_WORKERS.get("doorcam")

    def gen():
        prev = None
        while True:
            jpeg = worker.get_latest_jpeg(prev) if worker else None
            if jpeg is None:
                time.sleep(0.04)
                continue
            prev = jpeg
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/<cam_id>')
def stream_cam(cam_id):
    """Dynamic stream route for arbitrary camera IDs."""
    worker = CAMERA_WORKERS.get(cam_id)
    if not worker:
        return "Camera Not Found", 404

    def gen():
        prev = None
        while True:
            jpeg = worker.get_latest_jpeg(prev)
            if jpeg is None:
                continue
            prev = jpeg
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/ptz/move', methods=['POST'])
def ptz_move():
    worker = CAMERA_WORKERS.get("ptz_gate") or CAMERA_WORKERS.get("cpplus")
    if not worker:
        return jsonify({'status': 'error', 'msg': 'No PTZ camera found'}), 500
    data = request.json or {}
    worker.ptz_move(pan=data.get('pan', 0.0), tilt=data.get('tilt', 0.0))
    return jsonify({'status': 'ok'})

@app.route('/api/ptz/stop', methods=['POST'])
def ptz_stop():
    worker = CAMERA_WORKERS.get("ptz_gate") or CAMERA_WORKERS.get("cpplus")
    if not worker:
        return jsonify({'status': 'error'}), 500
    worker.ptz_stop()
    return jsonify({'status': 'ok'})

@app.route('/api/status', methods=['GET'])
def get_status():
    ptz_w = CAMERA_WORKERS.get("ptz_gate")
    door_w = CAMERA_WORKERS.get("doorcam")
    return jsonify({
        'motion': (ptz_w.motion_detected if ptz_w else False) or (door_w.motion_detected if door_w else False),
        'recording': (ptz_w.is_recording if ptz_w else False) or (door_w.is_recording if door_w else False),
        'auto_track': False
    })

@app.route('/api/recordings/dates', methods=['GET'])
def get_recording_dates():
    cam_filter = request.args.get('camera', 'all')
    all_files = glob.glob(os.path.join(EVENTS_DIR, "*.*"))
    dates = set()

    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in [".mp4", ".avi", ".mkv"] or os.path.basename(f).startswith("."):
            continue
        date_str, _, file_cam = parse_file_meta(f)
        if cam_filter in ['all', file_cam]:
            dates.add(date_str)

    return jsonify(sorted(list(dates)))

@app.route('/api/recordings/list', methods=['GET'])
def get_recordings_for_date():
    req_date = request.args.get('date')
    cam_filter = request.args.get('camera', 'all')
    if not req_date:
        return jsonify([])

    all_files = glob.glob(os.path.join(EVENTS_DIR, "*.*"))
    events = []

    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in [".mp4", ".avi", ".mkv"] or os.path.basename(f).startswith("."):
            continue
        date_str, time_str, file_cam = parse_file_meta(f)

        if date_str == req_date and (cam_filter in ['all', file_cam]):
            size_mb = round(os.path.getsize(f) / (1024 * 1024), 2)
            fname = os.path.basename(f)
            events.append({
                'filename': fname,
                'time': time_str,
                'camera': file_cam,
                'type': ext.replace(".", "").upper(),
                'size': f"{size_mb} MB",
                'url': f"/recordings/video/{fname}"
            })

    events.sort(key=lambda x: x['time'], reverse=True)
    return jsonify(events)

@app.route('/recordings/video/<filename>')
def serve_video(filename):
    file_path = os.path.join(EVENTS_DIR, filename)
    if not os.path.exists(file_path):
        return "File not found", 404

    base_name = os.path.splitext(filename)[0]
    cached_mp4 = os.path.join(CACHE_DIR, f"{base_name}.mp4")

    if not os.path.exists(cached_mp4):
        cmd = [
            "ffmpeg", "-y", "-i", file_path,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            cached_mp4
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return "Transcode failed", 500

    return send_file(cached_mp4, mimetype='video/mp4')

@app.route('/api/reload')
def api_reload():
    load_cameras()
    return jsonify(ok=True, cameras=list(CAMERA_WORKERS.keys()))

if __name__ == '__main__':
    load_cameras()
    app.run(host='0.0.0.0', port=5000, threaded=True)
