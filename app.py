#!/usr/bin/env python3
"""
FPV Ground Station - MAVLink WebSocket proxy + Flask backend
"""

import threading
import time
import os
import sys
import subprocess
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_socketio import SocketIO, emit

# Video capture
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: cv2 not available - video disabled")

# PyInstaller support — find bundled files
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MAVLink import
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("WARNING: pymavlink not available")

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))
app.config['SECRET_KEY'] = 'fpv-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global MAVLink connection and state
mav_connection = None
mav_lock = threading.Lock()
mav_connect_string = 'udpin:0.0.0.0:19856'
mav_status = {'connected': False, 'connecting': False, 'error': None, 'connection_string': 'udpin:0.0.0.0:19856'}
mav_manual_only = True  # No auto-connect on startup
_mav_conn_params = {'type': 'udpin', 'ip': '0.0.0.0', 'port': '19856', 'baud': '115200'}
mav_stop_event = threading.Event()

telemetry = {
    'roll': 0.0,
    'pitch': 0.0,
    'yaw': 0.0,
    'heading': 0,
    'airspeed': 0.0,
    'groundspeed': 0.0,
    'altitude_rel': 0.0,
    'altitude_amsl': 0.0,
    'battery_voltage': 0.0,
    'battery_remaining': -1,
    'gps_fix': 0,
    'gps_sats': 0,
    'armed': False,
    'flight_mode': 'UNKNOWN',
    'lat': 0.0,
    'lon': 0.0,
}

gps_disabled = False

# ArduPlane flight modes mapping
ARDUPLANE_MODES = {
    0: 'MANUAL',
    1: 'CIRCLE',
    2: 'STABILIZE',
    3: 'TRAINING',
    4: 'ACRO',
    5: 'FBWA',
    6: 'FBWB',
    7: 'CRUISE',
    8: 'AUTOTUNE',
    10: 'AUTO',
    11: 'RTL',
    12: 'LOITER',
    13: 'TAKEOFF',
    14: 'AVOID_ADSB',
    15: 'GUIDED',
    17: 'QSTABILIZE',
    18: 'QHOVER',
    19: 'QLOITER',
    20: 'QLAND',
    21: 'QRTL',
    22: 'QAUTOTUNE',
    23: 'QACRO',
    24: 'THERMAL',
    25: 'LOITER_ALT_QLAND',
}

MODE_NAME_TO_NUM = {v: k for k, v in ARDUPLANE_MODES.items()}

GPS_FIX_TYPES = {
    0: 'NO GPS',
    1: 'NO FIX',
    2: '2D FIX',
    3: '3D FIX',
    4: 'DGPS',
    5: 'RTK FLOAT',
    6: 'RTK FIXED',
}


def build_mavlink_connection(conn_type, ip, port, baud=115200):
    """Build pymavlink connection."""
    port = int(port)
    if conn_type in ('udpin', 'udpci'):
        # UDPCI = listen on local port, drone sends packets here
        # Same as Mission Planner UDPCI
        return mavutil.mavlink_connection(f'udpin:0.0.0.0:{port}',
                                          source_system=255, source_component=0)
    elif conn_type == 'udpout':
        # Send to remote IP
        return mavutil.mavlink_connection(f'udpout:{ip}:{port}',
                                          source_system=255, source_component=0)
    elif conn_type == 'tcp':
        return mavutil.mavlink_connection(f'tcp:{ip}:{port}',
                                          source_system=255, source_component=0)
    elif conn_type == 'serial':
        return mavutil.mavlink_connection(port, baud=baud,
                                          source_system=255, source_component=0)
    else:
        return mavutil.mavlink_connection(f'udpin:0.0.0.0:{port}',
                                          source_system=255, source_component=0)


def connect_mavlink(connection_string=None):
    """Connect to MAVLink — single attempt, no auto-retry."""
    global mav_connection, mav_connect_string, mav_status, mav_stop_event, \
           _mav_conn_params
    if connection_string:
        mav_connect_string = connection_string
    mav_stop_event.clear()
    try:
        cs = mav_connect_string
        print(f"Connecting to MAVLink: {cs}")
        mav_status['connecting'] = True
        mav_status['connected'] = False
        mav_status['error'] = None
        mav_status['connection_string'] = cs

        # Parse stored params for cleaner connection
        ct = _mav_conn_params.get('type', 'udpout')
        ip = _mav_conn_params.get('ip', '192.168.144.12')
        port = int(_mav_conn_params.get('port', 19856))
        baud = int(_mav_conn_params.get('baud', 115200))

        conn = build_mavlink_connection(ct, ip, port, baud)
        conn.wait_heartbeat(timeout=15)
        print(f"MAVLink connected: sys={conn.target_system}")
        with mav_lock:
            mav_connection = conn
        mav_status['connected'] = True
        mav_status['connecting'] = False
        mav_status['error'] = None
        receive_loop(conn)
    except Exception as e:
        print(f"MAVLink error: {e}")
        mav_status['error'] = str(e)
    finally:
        mav_status['connected'] = False
        mav_status['connecting'] = False
        if not mav_status.get('error'):
            mav_status['error'] = 'Disconnected'
        with mav_lock:
            mav_connection = None


def receive_loop(conn):
    """Receive and process MAVLink messages."""
    global telemetry
    import select as select_mod
    # For serial: flush stale buffer before starting
    try:
        if hasattr(conn, 'port') and hasattr(conn.port, 'flushInput'):
            conn.port.flushInput()
        # Drain any buffered packets quickly
        for _ in range(200):
            m = conn.recv_match(blocking=False)
            if m is None:
                break
    except Exception:
        pass

    while True:
        try:
            msg = conn.recv_match(blocking=True, timeout=5)
            if msg is None:
                continue
            msg_type = msg.get_type()

            if msg_type == 'ATTITUDE':
                import math
                telemetry['roll'] = math.degrees(msg.roll)
                telemetry['pitch'] = math.degrees(msg.pitch)
                telemetry['yaw'] = math.degrees(msg.yaw)

            elif msg_type == 'VFR_HUD':
                telemetry['airspeed'] = msg.airspeed
                telemetry['groundspeed'] = msg.groundspeed
                telemetry['heading'] = msg.heading
                telemetry['altitude_rel'] = msg.alt

            elif msg_type == 'GLOBAL_POSITION_INT':
                telemetry['lat'] = msg.lat / 1e7
                telemetry['lon'] = msg.lon / 1e7
                telemetry['altitude_amsl'] = msg.alt / 1000.0
                telemetry['altitude_rel'] = msg.relative_alt / 1000.0

            elif msg_type == 'GPS_RAW_INT':
                telemetry['gps_fix'] = msg.fix_type
                telemetry['gps_sats'] = msg.satellites_visible

            elif msg_type == 'SYS_STATUS':
                telemetry['battery_voltage'] = msg.voltage_battery / 1000.0
                telemetry['battery_remaining'] = msg.battery_remaining

            elif msg_type == 'HEARTBEAT':
                armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                telemetry['armed'] = armed
                mode_num = msg.custom_mode
                telemetry['flight_mode'] = ARDUPLANE_MODES.get(mode_num, f'MODE_{mode_num}')

        except Exception as e:
            print(f"MAVLink receive error: {e}")
            break


def emit_telemetry_loop():
    """Emit telemetry to all connected WebSocket clients every 200ms."""
    while True:
        try:
            socketio.emit('telemetry', {**telemetry, 'mav_status': mav_status})
        except Exception as e:
            pass
        time.sleep(0.2)


# ─── REST API ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/serial/ports', methods=['GET'])
def api_serial_ports():
    """List available serial/COM ports."""
    ports = []
    try:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            ports.append({
                'device': p.device,
                'description': p.description or p.device,
                'hwid': p.hwid or ''
            })
        ports.sort(key=lambda x: x['device'])
    except ImportError:
        # pyserial not installed - try manual detection
        import glob, platform
        if platform.system() == 'Windows':
            import winreg
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DEVICEMAP\SERIALCOMM')
                i = 0
                while True:
                    try:
                        name, val, _ = winreg.EnumValue(key, i)
                        ports.append({'device': val, 'description': val, 'hwid': ''})
                        i += 1
                    except OSError:
                        break
            except Exception:
                pass
        else:
            for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/tty.usbserial*', '/dev/tty.usbmodem*']:
                for p in glob.glob(pattern):
                    ports.append({'device': p, 'description': p, 'hwid': ''})
    except Exception as e:
        return jsonify({'ports': [], 'error': str(e)})
    return jsonify({'ports': ports})


@app.route('/api/connection', methods=['GET'])
def api_connection_status():
    return jsonify(mav_status)


@app.route('/api/connect', methods=['POST'])
def api_connect():
    global mav_connect_string, mav_stop_event, _mav_conn_params
    data = request.get_json() or {}
    ip   = data.get('ip', '').strip()
    port = str(data.get('port', '19856')).strip()
    conn_type = data.get('type', 'udpout')
    baud = str(data.get('baud', '115200'))

    if not ip and conn_type not in ('udpin', 'serial'):
        return jsonify({'error': 'IP is required'}), 400

    # Store params for cleaner connection building
    _mav_conn_params = {'type': conn_type, 'ip': ip, 'port': port, 'baud': baud}

    # Human-readable connection string for display only
    if conn_type in ('udpout', 'udpci'):
        cs = f'udpout:{ip}:{port}'
    elif conn_type == 'udpin':
        cs = f'udpin:0.0.0.0:{port}'
    elif conn_type == 'tcp':
        cs = f'tcp:{ip}:{port}'
    elif conn_type == 'serial':
        cs = f'{port}@{baud}'
    else:
        cs = f'udpout:{ip}:{port}'

    # Stop current connection
    mav_stop_event.set()
    time.sleep(0.3)

    mav_stop_event.clear()
    mav_connect_string = cs
    mav_status['connection_string'] = cs
    mav_status['connected'] = False
    mav_status['connecting'] = True
    mav_status['error'] = None

    t = threading.Thread(target=connect_mavlink, args=(cs,), daemon=True)
    t.start()

    return jsonify({'ok': True, 'connection_string': cs})


@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    global mav_stop_event
    mav_stop_event.set()
    with mav_lock:
        pass
    mav_status['connected'] = False
    mav_status['connecting'] = False
    mav_status['error'] = 'Disconnected'
    return jsonify({'ok': True})


@app.route('/api/arm', methods=['POST'])
def api_arm():
    data = request.get_json()
    arm = data.get('arm', True)
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    try:
        if arm:
            conn.arducopter_arm()
        else:
            conn.arducopter_disarm()
        return jsonify({'ok': True, 'armed': arm})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mode', methods=['POST'])
def api_mode():
    data = request.get_json()
    mode_name = data.get('mode', 'MANUAL')
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    mode_num = MODE_NAME_TO_NUM.get(mode_name)
    if mode_num is None:
        return jsonify({'error': f'Unknown mode: {mode_name}'}), 400
    try:
        conn.set_mode(mode_num)
        return jsonify({'ok': True, 'mode': mode_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/disable_gps', methods=['POST'])
def api_disable_gps():
    global gps_disabled
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    gps_disabled = not gps_disabled
    value = 0 if gps_disabled else 1
    try:
        conn.param_set_send('AHRS_GPS_USE', value, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
        return jsonify({'ok': True, 'gps_disabled': gps_disabled, 'AHRS_GPS_USE': value})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/restart_mission', methods=['POST'])
def api_restart_mission():
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    try:
        # Set current waypoint to 0
        conn.mav.mission_set_current_send(conn.target_system, conn.target_component, 0)
        # Set mode to AUTO
        auto_num = MODE_NAME_TO_NUM.get('AUTO', 10)
        conn.set_mode(auto_num)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── SocketIO events ──────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    print(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def on_disconnect():
    print(f"Client disconnected: {request.sid}")


# ─── Main ─────────────────────────────────────────────────────────────────────

# ─── Video Stream ─────────────────────────────────────────────────────────────
RTSP_URL = 'rtsp://192.168.144.25:8554/main.264'
GSTREAMER_PIPELINE = (
    f'rtspsrc location={RTSP_URL} latency=0 ! '
    'rtph264depay ! avdec_h264 ! videoconvert ! '
    'video/x-raw,format=BGR ! appsink name=outsink drop=true max-buffers=1'
)

video_frame = None
raw_frame = None  # Unencoded frame for tracker
frame_w = 640
frame_h = 480
video_lock = threading.Lock()
video_running = False
video_active = False  # Manual control - False = stopped

# ── Object Tracker ────────────────────────────────────────────────────────────
tracker = None
tracker_lock = threading.Lock()
tracker_bbox = None   # (x, y, w, h) normalized 0..1
tracker_active = False

def video_capture_loop():
    global video_frame, raw_frame, frame_w, frame_h, video_running, video_active
    global tracker, tracker_bbox, tracker_active
    video_running = True
    while video_running:
        if not video_active:
            time.sleep(0.5)
            continue
        cap = None
        try:
            if CV2_AVAILABLE:
                # Low-latency RTSP options via FFMPEG
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp|fflags;nobuffer|flags;low_delay|framedrop;1'
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
                    cap = cv2.VideoCapture(RTSP_URL)
                if not cap.isOpened():
                    print(f"Video: cannot open {RTSP_URL}")
                    video_active = False
                    socketio.emit('video_status', {'connected': False, 'error': 'Cannot connect to RTSP'})
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print("Video: connected")
                socketio.emit('video_status', {'connected': True, 'error': None, 'w': frame_w, 'h': frame_h})
                while video_running and video_active:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h, w = frame.shape[:2]
                    if w != frame_w or h != frame_h:
                        frame_w, frame_h = w, h
                        socketio.emit('video_dims', {'w': w, 'h': h})

                    # Run tracker
                    with tracker_lock:
                        if tracker_active and tracker is not None:
                            ok, bbox = tracker.update(frame)
                            if ok:
                                tx, ty, tw, th = [int(v) for v in bbox]
                                nb = {
                                    'x': tx / w, 'y': ty / h,
                                    'w': tw / w, 'h': th / h,
                                    'ok': True, 'active': True
                                }
                                tracker_bbox = nb
                                # Emit to frontend canvas
                                socketio.emit('tracker_box', nb)
                            else:
                                tracker_bbox = {'ok': False}
                                socketio.emit('tracker_box', {'ok': False, 'active': True})

                    # Save raw frame BEFORE encoding (for tracker init)
                    with video_lock:
                        raw_frame = frame.copy()
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    with video_lock:
                        video_frame = jpeg.tobytes()

                socketio.emit('video_status', {'connected': False, 'error': 'Stream lost'})
        except Exception as e:
            print(f"Video capture error: {e}")
            socketio.emit('video_status', {'connected': False, 'error': str(e)})
            video_active = False
        finally:
            if cap:
                cap.release()
            with video_lock:
                video_frame = None
                raw_frame = None
        time.sleep(1)


def gen_frames():
    while True:
        with video_lock:
            frame = video_frame
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)
        time.sleep(0.033)  # ~30fps cap


@app.route('/video')
def video_feed():
    return Response(stream_with_context(gen_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/video/start', methods=['POST'])
def api_video_start():
    global video_active, RTSP_URL
    data = request.get_json() or {}
    if data.get('url'):
        RTSP_URL = data['url']
    video_active = True
    return jsonify({'ok': True, 'url': RTSP_URL})

@app.route('/api/video/stop', methods=['POST'])
def api_video_stop():
    global video_active
    video_active = False
    with video_lock:
        pass
    return jsonify({'ok': True})

@app.route('/api/video/status', methods=['GET'])
def api_video_status():
    return jsonify({'active': video_active, 'url': RTSP_URL})


@app.route('/api/tracker/start', methods=['POST'])
def api_tracker_start():
    global tracker, tracker_active, tracker_bbox
    data = request.get_json() or {}
    # bbox in normalized coords 0..1
    nx = float(data.get('x', 0))
    ny = float(data.get('y', 0))
    nw = float(data.get('w', 0.1))
    nh = float(data.get('h', 0.1))
    with video_lock:
        frame = raw_frame
    if frame is None:
        return jsonify({'error': 'No video frame available'}), 400
    if not CV2_AVAILABLE:
        return jsonify({'error': 'cv2 not available'}), 400
    h, w = frame.shape[:2]
    x = int(nx * w)
    y = int(ny * h)
    bw = int(nw * w)
    bh = int(nh * h)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(10, min(bw, w - x))
    bh = max(10, min(bh, h - y))
    with tracker_lock:
        try:
            t = cv2.TrackerCSRT_create()
        except AttributeError:
            try:
                t = cv2.legacy.TrackerCSRT_create()
            except Exception:
                t = cv2.TrackerKCF_create()
        t.init(frame, (x, y, bw, bh))
        tracker = t
        tracker_active = True
        tracker_bbox = {'x': nx, 'y': ny, 'w': nw, 'h': nh, 'ok': True}
    return jsonify({'ok': True})


@app.route('/api/tracker/stop', methods=['POST'])
def api_tracker_stop():
    global tracker, tracker_active, tracker_bbox
    with tracker_lock:
        tracker = None
        tracker_active = False
        tracker_bbox = None
    socketio.emit('tracker_status', {'active': False})
    return jsonify({'ok': True})


if __name__ == '__main__':
    # Start video capture thread (waits for manual connect)
    if CV2_AVAILABLE:
        threading.Thread(target=video_capture_loop, daemon=True).start()
    else:
        print("cv2 not available - video disabled")

    # MAVLink: manual connect only — do NOT auto-connect on startup
    if not MAVLINK_AVAILABLE:
        print("MAVLink not available - running without telemetry")

    telem_thread = threading.Thread(target=emit_telemetry_loop, daemon=True)
    telem_thread.start()

    print("Starting FPV app on port 8086...")
    import webbrowser
    threading.Timer(2.0, lambda: webbrowser.open('http://localhost:8086')).start()
    socketio.run(app, host='0.0.0.0', port=8086, debug=False, allow_unsafe_werkzeug=True)
