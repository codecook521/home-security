# Vikirna Surveillance Hub (v1.0)

A unified, modular home security monitoring and pan-tilt-zoom (PTZ) controller built for Ubuntu servers. Vikirna Surveillance Hub manages mixed camera streams (RTSP and MJPEG), coordinates intelligent object and facial recognition pipelines, and records pristine event clips without bounding-box occlusions.

## Key Features

* **Decoupled Recording Pipeline:** Records raw, high-fidelity video clips without burning detection boxes or facial labels into the footage. Live detections remain exclusive to the preview interface.
* **Single-Consumer Proxy Architecture:** Alleviates buffer exhaustion and memory starvation on resource-constrained microcontrollers (such as the ESP32-CAM) by maintaining a single client ingest thread per source.
* **Modular Hardware Registry:** Camera streams, transport modes, framerate throttles, and ONVIF profiles are defined in an atomic configuration file (`cameras.json`).
* **Computer Vision Pipeline:** Real-time object detection via YOLOv8 paired with facial feature extraction and matching via DeepFace (`Facenet512`).
* **PTZ Control Integration:** Native ONVIF pan, tilt, and zoom command routing directly from the web dashboard.
* **Faststart Browser Playback:** Automatic background H.264 transcode containerization via FFmpeg (`+faststart`) for instant HTML5 video playback.
* **Calendar & History Browser:** Integrated dashboard allowing date-based filtering and instant modal playback of recorded incidents.

## System Architecture

```
                                  +-------------------+
                                  |   Web Dashboard   |
                                  |   (Flask / OSD)   |
                                  +---------+---------+
                                            ^
                                            | MJPEG Proxy Streams
+-----------------------+                   |
|   ESP32 Doorcam       +--- (HTTP/MJPEG) --+
|   (CAMERA_IP:81)   |                   |
+-----------------------+                   v
                               +-----------------------------+
+-----------------------+      |    modular_monitor.py       |
|   CP PLUS PTZ Gate    +----->|  - Worker Thread Manager    |
|   (RTSP / ONVIF)      |      |  - YOLOv8 & DeepFace        |
+-----------------------+      |  - VideoWriter (Pristine)   |
                               +--------------+--------------+
                                              |
                                              v (H.264 Transcode)
                                      +-------+-------+
                                      | /events/ dir  |
                                      +---------------+
```

## Directory Structure

```
~/home-security/
├── cameras.example.json      # Template configuration file
├── cameras.json              # Local active cameras (untracked in Git)
├── modular_monitor.py        # Core engine and web server
├── detections.csv            # Detection event log
├── vikirna_logo.png          # Video overlay watermark
├── LICENSE                   # MIT License file
├── templates/
│   └── index.html            # Surveillance web console
├── faces/                    # Reference portraits for DeepFace
│   ├── person1.jpg
│   └── person2.jpg
└── events/                   # Video storage directory
    └── .cache/               # Transcoded H.264 clips
```

## Prerequisites & Installation

### 1. System Dependencies

Ensure Python 3, FFmpeg, and OpenCV runtime tools are installed:

```bash
sudo apt update
sudo apt install -y python3 python3-pip ffmpeg libsm6 libxext6
```

### 2. Python Environment

Install the required computer vision and web server packages:

```bash
pip3 install flask opencv-python numpy Pillow ultralytics deepface onvif-zeep
```

## Configuration

Copy the example configuration file to create your active camera manifest:

```bash
cp cameras.example.json cameras.json
```

Edit `cameras.json` to reflect your network topology:

```json
[
  {
    "id": "doorcam",
    "name": "ESP32 Doorcam",
    "type": "mjpeg",
    "url": "http://<CAMERA_IP>:81/",
    "fps_limit": 6.0,
    "has_ptz": false
  },
  {
    "id": "ptz_gate",
    "name": "PTZ Gate Cam",
    "type": "rtsp",
    "url": "rtsp://<USER>:<PASS>@<CAMERA_IP>:5543/live/channel1",
    "fps_limit": 10.0,
    "has_ptz": true,
    "onvif": {
      "ip": "<CAMERA_IP>",
      "port": 8000,
      "user": "<USER>",
      "pass": "<PASS>"
    }
  }
]
```

*Note: Ensure `cameras.json` is kept private and never committed to source control.*

## Systemd Service Setup

To run the surveillance service automatically on boot:

1. Create the systemd unit file:
   ```bash
   sudo nano /etc/systemd/system/security-monitor.service
   ```

2. Add the service configuration:
   ```ini
   [Unit]
   Description=Vikirna Modular Surveillance Monitor & PTZ Controller
   After=network.target

   [Service]
   Type=simple
   User=username
   WorkingDirectory=/path/to/home-security
   ExecStart=/usr/bin/python3 /path/to/home-security/modular_monitor.py
   Restart=always
   RestartSec=5
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the daemon:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable security-monitor.service
   sudo systemctl start security-monitor.service
   ```

## Maintenance & Housekeeping

To prevent storage depletion, configure a daily cron job to delete recordings older than 7 days:

```bash
(crontab -l 2>/dev/null; echo "0 0 * * * find /path/to/home-security/events -type f -name '*.mp4' -mtime +7 -delete") | crontab -
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

* **Maintainer:** codecook521 ([username@gmail.com](mailto:username@gmail.com))
* **Version:** 1.0 (Stable)