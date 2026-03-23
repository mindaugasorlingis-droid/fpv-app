#!/usr/bin/env python3
"""
FPV Ground Station - MAVLink WebSocket proxy + Flask backend
"""

import threading
import time
import os
import sys
import subprocess
import numpy as np
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

# YOLO import (lazy — loaded on first use)
yolo_model = None
yolo_lock = threading.Lock()
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("WARNING: ultralytics not available - YOLO detection disabled")

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))
app.config['SECRET_KEY'] = 'fpv-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global MAVLink connection and state
mav_connection = None
mav_lock = threading.Lock()
mav_connect_string = 'udpin:0.0.0.0:19856'
mav_status = {'connected': False, 'connecting': False, 'error': None, 'connection_string': 'udpin:0.0.0.0:19856'}
mav_manual_only = True  # No auto-connect on startup
mav_stop_event = threading.Event()

# Mission Planner WebSocket bridge mode
# When enabled, instead of direct UDP/serial, we connect to MP's ws://host:56781/websocket/raw
# and pipe raw MAVLink bytes through it
MP_WS_URL = None   # e.g. 'ws://192.168.1.100:56781/websocket/raw' — set via UI
mp_ws = None       # websocket connection
mp_ws_lock = threading.Lock()

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
    'throttle': 0,
    # Radio link quality (RADIO_STATUS / RC_CHANNELS RSSI)
    'rssi': None,        # local RSSI raw (0-254)
    'remrssi': None,     # remote RSSI raw (0-254)
    'rssi_dbm': None,    # local RSSI dBm (converted)
    'remrssi_dbm': None, # remote RSSI dBm
    'rxerrors': None,    # packet errors
    'noise': None,       # noise floor raw
    'radio_source': None,# 'RADIO_STATUS' or 'RC_CHANNELS'
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


def make_connection(cs):
    """Build MAVLink connection.

    UDPCI (Mission Planner UdpSerial style):
      - Bind to local port, wait for drone to send first packet
      - Learn drone's IP:port from first received packet
      - Send all replies back to that endpoint
      - No connect() call — pure server/listen socket
      - Uses blocking socket with timeout (no setblocking(False) issues on Windows)

    UDP out:
      - Windows-safe: create socket, connect() while blocking, then setblocking(False)
    """
    import socket as _socket

    if cs.startswith('udpin:'):
        # UDPCI mode — exactly like Mission Planner UdpSerial.Open()
        parts = cs.split(':', 2)
        local_port = int(parts[2])
        print(f"MAVLink UDPCI: binding to 0.0.0.0:{local_port}, waiting for drone...")

        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)  # 1s timeout so reconnect loop can check mav_stop_event
        sock.bind(('0.0.0.0', local_port))

        # Wait for first packet from drone (like MP: while BytesToRead==0)
        remote_addr = None
        while remote_addr is None:
            if mav_stop_event.is_set():
                sock.close()
                raise Exception("Cancelled")
            try:
                data, addr = sock.recvfrom(280)
                remote_addr = addr
                print(f"MAVLink UDPCI: drone found at {addr[0]}:{addr[1]}")
            except _socket.timeout:
                continue

        # Use non-blocking so pymavlink recv() returns "" instead of raising socket.timeout
        # pymavlink handles EAGAIN/EWOULDBLOCK correctly — socket.timeout it does NOT
        sock.setblocking(False)

        # Build mavudp server-style: knows remote endpoint
        conn = mavutil.mavudp.__new__(mavutil.mavudp)
        conn.port = sock
        conn.udp_server = True
        conn.broadcast = False
        conn.destination_addr = remote_addr
        conn.resolved_destination_addr = remote_addr[0]
        conn.last_address = remote_addr
        conn.timeout = 0
        conn.clients = set([remote_addr])
        conn.clients_last_alive = {remote_addr: time.time()}
        mavutil.mavfile.__init__(conn, sock.fileno(),
                                 f'udpin:{remote_addr[0]}:{remote_addr[1]}',
                                 source_system=255, source_component=0,
                                 input=False)
        # Parse first packet — this may set target_system if it's a HEARTBEAT
        try:
            conn.mav.parse_buffer(data)
        except Exception:
            pass
        return conn

    elif cs.startswith('udpout:') or cs.startswith('udp:'):
        parts = cs.split(':', 2)
        host = parts[1].lstrip('/')
        port = int(parts[2])

        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        sock.bind(('', 0))
        sock.connect((host, port))
        sock.setblocking(False)

        conn = mavutil.mavudp.__new__(mavutil.mavudp)
        conn.port = sock
        conn.udp_server = False
        conn.broadcast = False
        conn.destination_addr = (host, port)
        conn.resolved_destination_addr = host
        conn.last_address = None
        conn.timeout = 0
        conn.clients = set()
        conn.clients_last_alive = {}
        mavutil.mavfile.__init__(conn, sock.fileno(), f'{host}:{port}',
                                 source_system=255, source_component=0,
                                 input=False)
        return conn

    else:
        # TCP, serial, bluetooth
        return mavutil.mavlink_connection(cs, source_system=255, source_component=0)


def connect_mp_websocket(ws_url):
    """Connect to Mission Planner via ws://host:56781/websocket/raw

    MP streams raw MAVLink bytes over WebSocket binary frames.
    We build a MAVLink parser directly on top — no UDP loopback needed.
    This avoids all the UDP race conditions and select() issues.

    Architecture:
      WebSocket recv thread → byte buffer → MAVLink parser (in-thread)
      MAVLink send → WebSocket send_binary
    """
    import websocket as _ws
    import queue
    from pymavlink.dialects.v20 import ardupilotmega as mav_dialect
    global mav_connection, mav_status, mav_stop_event, mp_ws

    print(f"MP WebSocket: connecting to {ws_url}")
    mav_status['connecting'] = True
    mav_status['connected'] = False
    mav_status['error'] = None
    mav_status['connection_string'] = ws_url

    ws_conn = None

    try:
        ws_conn = _ws.create_connection(ws_url, timeout=10)
        with mp_ws_lock:
            mp_ws = ws_conn
        print("MP WebSocket: connected to MP")

        # Build MAVLink connection using TCP loopback as transport
        # but use mavudp in server mode with our own socket pair
        import socket as _socket
        import types

        # Create a proper UDP server socket that pymavlink can use
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.bind(('127.0.0.1', 0))
        sock.setblocking(False)
        lport = sock.getsockname()[1]

        # Sender — injects WS data into our socket
        sender = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sender.connect(('127.0.0.1', lport))

        # Build mavudp manually so we control the socket
        conn = mavutil.mavudp.__new__(mavutil.mavudp)
        conn.port = sock
        conn.udp_server = True
        conn.broadcast = False
        conn.destination_addr = ('127.0.0.1', lport)
        conn.resolved_destination_addr = '127.0.0.1'
        conn.last_address = None
        conn.timeout = 0
        conn.clients = set()
        conn.clients_last_alive = {}
        mavutil.mavfile.__init__(conn, sock.fileno(),
                                 f'mp_ws:{ws_url}',
                                 source_system=255, source_component=0,
                                 input=False)

        # Override write → WebSocket
        def _write(self, buf):
            try:
                with mp_ws_lock:
                    if mp_ws and mp_ws.connected:
                        mp_ws.send_binary(bytes(buf))
            except Exception as e:
                print(f"MP WS send error: {e}")
        conn.write = types.MethodType(_write, conn)

        # WebSocket → UDP inject thread
        ws_dead = threading.Event()
        def ws_reader():
            buf = b''
            while not mav_stop_event.is_set() and not ws_dead.is_set():
                try:
                    ws_conn.settimeout(2.0)
                    data = ws_conn.recv()
                    if data is None:
                        continue
                    if isinstance(data, str):
                        data = data.encode('latin-1')
                    if data:
                        # Inject in chunks — UDP has size limits
                        # but MAVLink packets are small (<280 bytes)
                        # MP may bundle multiple packets per WS frame
                        # Feed byte by byte to avoid split packets via UDP size limit
                        chunk_size = 256
                        for i in range(0, len(data), chunk_size):
                            try:
                                sender.send(data[i:i+chunk_size])
                            except Exception:
                                pass
                except _ws.WebSocketTimeoutException:
                    continue
                except Exception as e:
                    print(f"MP WS recv error: {e}")
                    ws_dead.set()
                    break
            try: sender.close()
            except: pass

        t = threading.Thread(target=ws_reader, daemon=True)
        t.start()

        # Wait for first packet so pymavlink learns remote addr
        print("Waiting for first MAVLink packet from MP...")
        deadline = time.time() + 30
        while not ws_dead.is_set() and time.time() < deadline:
            if mav_stop_event.is_set():
                raise Exception("Cancelled")
            try:
                m = conn.recv_match(blocking=False)
                if m is not None:
                    print(f"First packet: {m.get_type()}")
                    if conn.target_system != 0:
                        break
            except Exception:
                pass
            time.sleep(0.02)

        if ws_dead.is_set():
            raise Exception("WebSocket closed before heartbeat")
        if conn.target_system == 0:
            raise Exception("No heartbeat from MP in 30s")

        print(f"MP MAVLink ready: sys={conn.target_system} comp={conn.target_component}")

        conn.mav.request_data_stream_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
        )

        with mav_lock:
            mav_connection = conn
        mav_status['connected'] = True
        mav_status['connecting'] = False
        mav_status['error'] = None
        socketio.emit('mavlink_status', {'connected': True, 'error': None, 'mode': 'MP'})

        # Run receive loop — exits when ws_dead or mav_stop_event
        receive_loop_mp(conn, ws_dead)

    except Exception as e:
        print(f"MP WebSocket error: {e}")
        mav_status['error'] = str(e)
    finally:
        mav_status['connected'] = False
        mav_status['connecting'] = False
        with mav_lock:
            mav_connection = None
        with mp_ws_lock:
            mp_ws = None
        try: ws_conn and ws_conn.close()
        except: pass
        socketio.emit('mavlink_status', {'connected': False, 'error': mav_status.get('error', 'Disconnected')})


def receive_loop_mp(conn, ws_dead):
    """receive_loop variant for MP WebSocket — also checks ws_dead event."""
    import select as _select
    global telemetry
    last_heartbeat = time.time()
    while not ws_dead.is_set() and not mav_stop_event.is_set():
        try:
            try:
                fd = conn.port.fileno()
                r, _, _ = _select.select([fd], [], [], 1.0)
            except Exception:
                r = [True]
            if not r:
                if time.time() - last_heartbeat > 15:
                    print("MP MAVLink: no heartbeat 15s")
                    break
                continue
            msg = conn.recv_match(blocking=False)
            if msg is None:
                continue
            # reuse same message processing as receive_loop
            # by injecting into the shared receive path
            _handle_mavlink_msg(msg, conn)
            if msg.get_type() == 'HEARTBEAT':
                last_heartbeat = time.time()
        except Exception as e:
            print(f"MP receive error: {e}")
            break


def connect_mavlink(connection_string=None):
    """Connect to MAVLink — auto-reconnects on disconnect."""
    global mav_connection, mav_connect_string, mav_status, mav_stop_event

    if connection_string:
        mav_connect_string = connection_string

    mav_stop_event.clear()

    while not mav_stop_event.is_set():
        cs = mav_connect_string

        # MP WebSocket bridge mode
        if cs.startswith('mp:') or cs.startswith('ws://') or cs.startswith('wss://'):
            ws_url = cs[3:] if cs.startswith('mp:') else cs
            connect_mp_websocket(ws_url)
            if mav_stop_event.is_set():
                break
            print("MP WebSocket disconnected — reconnecting in 3s...")
            mav_status['error'] = 'Reconnecting...'
            time.sleep(3)
            continue

        print(f"Connecting to MAVLink: {cs}")
        mav_status['connecting'] = True
        mav_status['connected'] = False
        mav_status['error'] = None
        mav_status['connection_string'] = cs
        conn = None
        try:
            conn = make_connection(cs)

            # Wait until we know target_system (need a HEARTBEAT from vehicle)
            # make_connection(udpin) already has first packet, but may not be HEARTBEAT
            # For non-udpin we explicitly wait
            print("Waiting for vehicle heartbeat...")
            deadline = time.time() + 30
            while conn.target_system == 0 and time.time() < deadline:
                if mav_stop_event.is_set():
                    raise Exception("Cancelled")
                m = conn.recv_match(type='HEARTBEAT', blocking=False)
                if m is None:
                    time.sleep(0.05)
            if conn.target_system == 0:
                raise Exception("No heartbeat received in 30s")
            print(f"MAVLink connected: sys={conn.target_system} comp={conn.target_component}")

            # Request all telemetry streams at 10Hz (like Mission Planner)
            conn.mav.request_data_stream_send(
                conn.target_system, conn.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
            )

            with mav_lock:
                mav_connection = conn
            mav_status['connected'] = True
            mav_status['connecting'] = False
            mav_status['error'] = None
            socketio.emit('mavlink_status', {'connected': True, 'error': None})

            receive_loop(conn)  # blocks until disconnect

        except Exception as e:
            print(f"MAVLink error: {e}")
            mav_status['error'] = str(e)
        finally:
            mav_status['connected'] = False
            mav_status['connecting'] = False
            with mav_lock:
                mav_connection = None
            if conn:
                try: conn.close()
                except Exception: pass
            socketio.emit('mavlink_status', {'connected': False, 'error': mav_status.get('error', 'Disconnected')})

        if mav_stop_event.is_set():
            break
        print("MAVLink disconnected — reconnecting in 3s...")
        mav_status['error'] = 'Reconnecting...'
        time.sleep(3)


def _handle_mavlink_msg(msg, conn):
    """Process a single MAVLink message — shared by receive_loop and receive_loop_mp."""
    global telemetry, guide_active
    import math as _math
    msg_type = msg.get_type()
    if msg_type == 'ATTITUDE':
        telemetry['roll']  = _math.degrees(msg.roll)
        telemetry['pitch'] = _math.degrees(msg.pitch)
        telemetry['yaw']   = _math.degrees(msg.yaw)
    elif msg_type == 'VFR_HUD':
        telemetry['airspeed']    = msg.airspeed
        telemetry['groundspeed'] = msg.groundspeed
        telemetry['heading']     = msg.heading
        telemetry['altitude_rel']= msg.alt
        telemetry['throttle']    = msg.throttle
    elif msg_type == 'GLOBAL_POSITION_INT':
        telemetry['lat']          = msg.lat / 1e7
        telemetry['lon']          = msg.lon / 1e7
        telemetry['altitude_amsl']= msg.alt / 1000.0
        telemetry['altitude_rel'] = msg.relative_alt / 1000.0
    elif msg_type == 'GPS_RAW_INT':
        telemetry['gps_fix']  = msg.fix_type
        telemetry['gps_sats'] = msg.satellites_visible
    elif msg_type == 'SYS_STATUS':
        telemetry['battery_voltage']   = msg.voltage_battery / 1000.0
        telemetry['battery_remaining'] = msg.battery_remaining
    elif msg_type == 'RADIO_STATUS':
        # SIYI / SiK radio link quality
        # MAVLink RADIO_STATUS rssi: 0=no signal, 254=max, 255=unknown
        # SiK radios: dBm = rssi / 1.9 - 127 (approximately)
        # SIYI airunit reports raw RSSI; conversion: dBm ≈ (rssi - 256) if rssi > 127 else rssi
        def raw_to_dbm(raw):
            if raw is None or raw == 255:
                return None
            # Standard SiK/SIYI conversion: rssi_dbm = raw - 256 (signed byte interpretation)
            if raw > 127:
                return raw - 256
            return raw
        rssi_raw   = getattr(msg, 'rssi', 255)
        remrssi_raw= getattr(msg, 'remrssi', 255)
        telemetry['rssi']       = rssi_raw
        telemetry['remrssi']    = remrssi_raw
        telemetry['rssi_dbm']   = raw_to_dbm(rssi_raw)
        telemetry['remrssi_dbm']= raw_to_dbm(remrssi_raw)
        telemetry['rxerrors']   = getattr(msg, 'rxerrors', 0)
        telemetry['noise']      = getattr(msg, 'noise', 0)
        telemetry['radio_source'] = 'RADIO_STATUS'
    elif msg_type == 'HEARTBEAT':
        armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        telemetry['armed'] = armed
        mode_num = msg.custom_mode
        telemetry['flight_mode'] = ARDUPLANE_MODES.get(mode_num, f'MODE_{mode_num}')
    elif msg_type == 'STATUSTEXT':
        text = msg.text.rstrip('\x00').strip()
        if text:
            severity_labels = {0:'EMERGENCY',1:'ALERT',2:'CRITICAL',3:'ERROR',
                               4:'WARNING',5:'NOTICE',6:'INFO',7:'DEBUG'}
            socketio.emit('mavlink_msg', {
                'text': text, 'severity': msg.severity,
                'level': severity_labels.get(msg.severity, 'INFO'), 'ts': time.time()
            })
    elif msg_type == 'EKF_STATUS_REPORT':
        socketio.emit('ekf_status', {'flags': msg.flags})
    elif msg_type == 'RC_CHANNELS':
        # RSSI fallback — use only if RADIO_STATUS not already providing data
        if telemetry.get('radio_source') != 'RADIO_STATUS':
            rc_rssi = getattr(msg, 'rssi', 255)
            if rc_rssi != 255:
                # RC_CHANNELS RSSI: 0-254, 255=unknown. Scale to dBm approx.
                # ArduPilot maps 0-254 → roughly -120..-20 dBm
                telemetry['rssi']     = rc_rssi
                telemetry['rssi_dbm'] = round(-120 + (rc_rssi / 254.0) * 100) if rc_rssi < 255 else None
                telemetry['radio_source'] = 'RC_CHANNELS'
        if guide_active:
            RC_DEADZONE = 100
            ch1 = getattr(msg, 'chan1_raw', 1500)
            ch2 = getattr(msg, 'chan2_raw', 1500)
            if abs(ch1 - 1500) > RC_DEADZONE or abs(ch2 - 1500) > RC_DEADZONE:
                print(f"RC takeover: ch1={ch1} ch2={ch2}")
                guide_active = False
                socketio.emit('guide_status', {'active': False, 'mode': 'OFF', 'reason': 'RC takeover'})
                socketio.emit('mavlink_msg', {
                    'text': 'RC takeover - GUIDE stopped, switching to FBWA',
                    'severity': 4, 'level': 'WARNING', 'ts': time.time()
                })
                try:
                    conn.set_mode(MODE_NAME_TO_NUM.get('FBWA', 5))
                except Exception:
                    pass


def receive_loop(conn):
    """Receive and process MAVLink messages.

    Uses non-blocking recv + select() so we never block indefinitely.
    pymavlink handles EAGAIN/EWOULDBLOCK correctly; socket.timeout it does NOT.
    """
    import select as _select
    global telemetry
    # For serial: flush stale buffer before starting
    try:
        if hasattr(conn, 'port') and hasattr(conn.port, 'flushInput'):
            conn.port.flushInput()
    except Exception:
        pass

    last_heartbeat = time.time()
    while True:
        try:
            # Wait up to 1s for data (select avoids busy-spin without blocking forever)
            try:
                fd = conn.port.fileno() if hasattr(conn, 'port') else conn.fd
                r, _, _ = _select.select([fd], [], [], 1.0)
            except Exception:
                r = [True]  # fallback — just try recv

            if not r:
                # 1s passed, no data — check heartbeat timeout
                if time.time() - last_heartbeat > 15:
                    print("MAVLink: no heartbeat for 15s — reconnecting")
                    break
                continue

            msg = conn.recv_match(blocking=False)
            if msg is None:
                continue
            _handle_mavlink_msg(msg, conn)
            if msg.get_type() == 'HEARTBEAT':
                last_heartbeat = time.time()

        except Exception as e:
            print(f"MAVLink receive error: {e}")
            break


def gcs_heartbeat_loop():
    """Send GCS heartbeat every 1s — required for drone to accept commands."""
    while True:
        try:
            with mav_lock:
                conn = mav_connection
            if conn:
                conn.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                )
        except Exception:
            pass
        time.sleep(1)


def emit_telemetry_loop():
    """Emit telemetry to all connected WebSocket clients every 200ms."""
    while True:
        try:
            socketio.emit('telemetry', {**telemetry, 'mav_status': mav_status})
            socketio.emit('guide_status', {'active': guide_active, 'mode': 'GUIDED' if guide_active else 'OFF'})
        except Exception as e:
            pass
        time.sleep(0.2)


# ─── GUIDE MODE LOGIC ────────────────────────────────────────────────────────

def guidance_loop():
    """Background guidance loop: reads tracker_bbox, sends MAVLink attitude commands."""
    global guide_active
    import math

    GUIDED_MODE_NUM = MODE_NAME_TO_NUM.get('GUIDED', 15)
    last_guided_send = 0.0      # time of last GUIDED mode request
    tracker_lost_time = None    # when we noticed tracker was gone

    print("GUIDE: guidance loop started")

    while guide_active:
        now = time.time()

        with mav_lock:
            conn = mav_connection

        if not conn:
            time.sleep(0.1)
            continue

        # Send GUIDED mode request at 1 Hz to keep it active
        if now - last_guided_send >= 1.0:
            try:
                conn.set_mode(GUIDED_MODE_NUM)
            except Exception:
                pass
            last_guided_send = now

        # Read current tracker bbox
        with tracker_lock:
            bbox = tracker_bbox

        # Check for tracker lost
        if bbox is None or not bbox.get('ok', True):
            if tracker_lost_time is None:
                tracker_lost_time = now
            elif now - tracker_lost_time >= 2.0:
                print("GUIDE: tracker lost for 2s — stopping guidance")
                guide_active = False
                break
            time.sleep(0.1)
            continue
        else:
            tracker_lost_time = None

        # Compute azimuth/elevation errors
        cx = bbox['x'] + bbox['w'] / 2.0
        cy = bbox['y'] + bbox['h'] / 2.0
        az_error = (cx - 0.5) * CAM_FOV_H   # + = right
        el_error = (0.5 - cy) * CAM_FOV_V   # + = up

        # Dead zone — on axis, no command needed
        if abs(az_error) < 1.0 and abs(el_error) < 1.0:
            time.sleep(0.1)
            continue

        # Proportional navigation
        Kp_roll  = 0.02   # rad per degree of azimuth error
        Kp_pitch = 0.015  # rad per degree of elevation error

        roll_cmd  = math.radians(az_error * Kp_roll * (180.0 / math.pi))
        pitch_cmd = math.radians(el_error * Kp_pitch * (180.0 / math.pi))

        # Clamp to safe angles
        roll_cmd  = max(-0.5, min(0.5, roll_cmd))
        pitch_cmd = max(-0.3, min(0.3, pitch_cmd))

        # Build quaternion from roll/pitch/yaw=0
        # q = [w, x, y, z]
        cr = math.cos(roll_cmd  / 2.0)
        sr = math.sin(roll_cmd  / 2.0)
        cp = math.cos(pitch_cmd / 2.0)
        sp = math.sin(pitch_cmd / 2.0)
        q = [cr * cp, sr * cp, cr * sp, -sr * sp]

        # type_mask: 0b00000111 = ignore body rates
        type_mask = 0b00000111

        try:
            conn.mav.set_attitude_target_send(
                int((now % 1000) * 1000),   # time_boot_ms
                conn.target_system,
                conn.target_component,
                type_mask,
                [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                0.0,   # roll rate
                0.0,   # pitch rate
                0.0,   # yaw rate
                0.5    # thrust (neutral)
            )
        except Exception as e:
            print(f"GUIDE: MAVLink send error: {e}")

        time.sleep(0.1)  # 10 Hz guidance loop

    guide_active = False
    print("GUIDE: guidance loop stopped")


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
    """Accept a raw connection_string and connect directly via pymavlink."""
    global mav_connect_string, mav_stop_event
    data = request.get_json() or {}

    # Accept raw connection_string field
    cs = data.get('connection_string', '').strip()
    if not cs:
        return jsonify({'error': 'connection_string is required'}), 400

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

    # ArduPlane 4.1+: GPS_TYPE=0 disables GPS driver entirely
    # ArduPlane <4.1: AHRS_GPS_USE=0 disables GPS fusion
    # We send both to cover all versions
    results = {}
    errors = []
    try:
        if gps_disabled:
            # Disable GPS: GPS_TYPE=0 (no GPS), AHRS_GPS_USE=0
            conn.param_set_send('GPS_TYPE',    0, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
            conn.param_set_send('AHRS_GPS_USE', 0, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
            results = {'GPS_TYPE': 0, 'AHRS_GPS_USE': 0}
        else:
            # Re-enable GPS: GPS_TYPE=1 (auto), AHRS_GPS_USE=1
            conn.param_set_send('GPS_TYPE',    1, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
            conn.param_set_send('AHRS_GPS_USE', 1, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
            results = {'GPS_TYPE': 1, 'AHRS_GPS_USE': 1}
        return jsonify({'ok': True, 'gps_disabled': gps_disabled, 'params': results})
    except Exception as e:
        gps_disabled = not gps_disabled  # rollback toggle
        return jsonify({'error': str(e)}), 500


@app.route('/api/restart_mission', methods=['POST'])
def api_restart_mission():
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    try:
        conn.mav.mission_set_current_send(conn.target_system, conn.target_component, 0)
        auto_num = MODE_NAME_TO_NUM.get('AUTO', 10)
        conn.set_mode(auto_num)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rc_override', methods=['POST'])
def api_rc_override():
    data = request.get_json() or {}
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    try:
        roll     = int(data.get('roll', 1500))
        pitch    = int(data.get('pitch', 1500))
        throttle = int(data.get('throttle', 1500))
        yaw      = int(data.get('yaw', 1500))
        # RC_CHANNELS_OVERRIDE: channels 1-8 (0=ignore)
        conn.mav.rc_channels_override_send(
            conn.target_system, conn.target_component,
            roll, pitch, throttle, yaw,
            0, 0, 0, 0  # ch5-ch8 ignore
        )
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preflight_calibration', methods=['POST'])
def api_preflight_calibration():
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    try:
        # MAV_CMD_PREFLIGHT_CALIBRATION (241)
        # param3=1: airspeed/pressure sensor calibration
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_CALIBRATION,
            0,  # confirmation
            0,  # param1: gyro
            0,  # param2: magnetometer
            1,  # param3: airspeed/pressure calibration
            0,  # param4: radio
            0,  # param5: accelerometer
            0, 0
        )
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preflight_reboot', methods=['POST'])
def api_preflight_reboot():
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    try:
        # MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN (246)
        # param1=1: reboot autopilot
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/preflight_shutdown', methods=['POST'])
def api_preflight_shutdown():
    with mav_lock:
        conn = mav_connection
    if not conn:
        return jsonify({'error': 'MAVLink not connected'}), 503
    try:
        # param1=2: shutdown autopilot
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0, 2, 0, 0, 0, 0, 0, 0
        )
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
RTSP_URL = 'rtsp://192.168.144.25:8554/main.264'  # SIYI camera
# GStreamer pipeline mode: set RTSP_URL to a pipeline string starting with "gst:"
# Example: gst:rtspsrc location=rtsp://192.168.144.25:8554/main.264 latency=0 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink
# This runs gst-launch-1.0 and pipes frames via stdout (fdsink + rawvideo)

video_frame = None
raw_frame = None  # Unencoded frame for tracker
raw_frame_event = threading.Event()  # Signals when raw_frame becomes available
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

# ── GUIDE MODE ────────────────────────────────────────────────────────────────
guide_active = False
guide_thread = None
CAM_FOV_H = 90.0   # Horizontal FOV degrees
CAM_FOV_V = 67.0   # Vertical FOV degrees


class OpticalFlowTracker:
    """Robust optical flow tracker with adaptive re-detection."""
    def __init__(self, frame, bbox):
        self.bbox = list(bbox)  # [x, y, w, h] pixels
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = [int(v) for v in bbox]
        roi = gray[y:y+h, x:x+w]
        pts = cv2.goodFeaturesToTrack(roi, maxCorners=100,
                                       qualityLevel=0.01, minDistance=3)
        if pts is not None:
            pts[:, :, 0] += x
            pts[:, :, 1] += y
            self.pts = pts
        else:
            # Fallback: grid of points in bbox
            grid = []
            for gx in range(x+5, x+w-5, 8):
                for gy in range(y+5, y+h-5, 8):
                    grid.append([[float(gx), float(gy)]])
            self.pts = np.array(grid, dtype=np.float32) if grid else None
        self.prev_gray = gray
        self.lost_frames = 0

    def update(self, frame):
        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.pts is None or len(self.pts) < 3:
            self.prev_gray = gray
            self.lost_frames += 1
            return False, self.bbox

        # Track points
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        good_new = pts2[status.flatten() == 1].reshape(-1, 2)
        good_old = self.pts[status.flatten() == 1].reshape(-1, 2)

        if len(good_new) < 3:
            self.lost_frames += 1
            self.prev_gray = gray
            # Try re-detect in expanded area
            x, y, w, h = [int(v) for v in self.bbox]
            pad = int(max(w, h) * 0.5)
            rx = max(0, x - pad)
            ry = max(0, y - pad)
            rw = w + pad * 2
            rh = h + pad * 2
            roi = gray[ry:ry+rh, rx:rx+rw]
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=50,
                                           qualityLevel=0.01, minDistance=4)
            if pts is not None:
                pts[:, :, 0] += rx
                pts[:, :, 1] += ry
                self.pts = pts
                self.lost_frames = 0
            return self.lost_frames <= 5, self.bbox

        # Compute translation
        dx = float(np.median(good_new[:, 0] - good_old[:, 0]))
        dy = float(np.median(good_new[:, 1] - good_old[:, 1]))

        # Update bbox
        self.bbox[0] += dx
        self.bbox[1] += dy

        # Keep bbox in frame
        fh, fw = frame.shape[:2]
        self.bbox[0] = max(0, min(self.bbox[0], fw - self.bbox[2]))
        self.bbox[1] = max(0, min(self.bbox[1], fh - self.bbox[3]))

        self.pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = gray
        self.lost_frames = 0

        # Periodically refresh feature points
        if len(self.pts) < 10:
            x, y, w, h = [int(v) for v in self.bbox]
            roi = gray[y:y+h, x:x+w]
            pts = cv2.goodFeaturesToTrack(roi, maxCorners=80,
                                           qualityLevel=0.01, minDistance=3)
            if pts is not None:
                pts[:, :, 0] += x
                pts[:, :, 1] += y
                self.pts = pts

        return True, tuple(self.bbox)

# ── YOLO detections ───────────────────────────────────────────────────────────
yolo_detections = []          # list of {x, y, w, h, conf} normalized 0..1
yolo_detect_enabled = False   # toggled by client
yolo_detections_lock = threading.Lock()


def get_yolo_model():
    """Lazy-load YOLOv8n model on first use."""
    global yolo_model
    with yolo_lock:
        if yolo_model is None and YOLO_AVAILABLE:
            print("Loading YOLOv8n model (downloading if needed)...")
            yolo_model = YOLO('yolov8n.pt')
            print("YOLOv8n model loaded.")
        return yolo_model


def yolo_detection_loop():
    """Background thread: run YOLO person detection every 500ms when tracker is NOT active."""
    global yolo_detections
    while True:
        time.sleep(0.5)
        if not yolo_detect_enabled:
            continue
        if tracker_active:
            # Don't run detection when tracker is running
            continue
        if not YOLO_AVAILABLE:
            continue

        with video_lock:
            frame = raw_frame.copy() if raw_frame is not None else None

        if frame is None:
            continue

        try:
            model = get_yolo_model()
            if model is None:
                continue

            h, w = frame.shape[:2]
            results = model(frame, classes=[0], verbose=False)  # class 0 = person
            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.35:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    dets.append({
                        'x': x1 / w,
                        'y': y1 / h,
                        'w': (x2 - x1) / w,
                        'h': (y2 - y1) / h,
                        'conf': round(conf, 2)
                    })

            with yolo_detections_lock:
                yolo_detections = dets

            socketio.emit('detections', dets)

        except Exception as e:
            print(f"YOLO detection error: {e}")


def tracker_loop():
    """Runs tracker in its own thread — decoupled from video capture.
    Reads raw_frame, runs NanoTrack update, emits result.
    This way video capture never blocks waiting for tracker inference.
    """
    global tracker_bbox, tracker_active, tracker
    import time as _time

    while True:
        # Wait for a new frame (max 100ms)
        got = raw_frame_event.wait(timeout=0.1)
        if not got:
            continue

        with tracker_lock:
            if not tracker_active or tracker is None:
                time.sleep(0.01)
                continue

        # Grab current frame (fast, under lock)
        with video_lock:
            frame = raw_frame
            raw_frame_event.clear()

        if frame is None:
            continue

        # Make a copy so video_capture_loop can overwrite raw_frame safely
        frame = frame.copy()
        h, w = frame.shape[:2]
        if w == 0 or h == 0:
            continue

        with tracker_lock:
            if not tracker_active or tracker is None:
                continue
            try:
                ok, bbox = tracker.update(frame)
            except Exception as e:
                print(f"Tracker error: {e}")
                tracker_active = False
                tracker_bbox = {'ok': False}
                socketio.emit('tracker_box', {'ok': False, 'active': False})
                continue

        if ok:
            tx, ty, tw, th = [int(v) for v in bbox]
            nb = {
                'x': tx / w, 'y': ty / h,
                'w': tw / w, 'h': th / h,
                'ok': True, 'active': True
            }
            tracker_bbox = nb
            socketio.emit('tracker_box', nb)
        else:
            tracker_bbox = {'ok': False}
            socketio.emit('tracker_box', {'ok': False, 'active': True})


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
                url = RTSP_URL
                if url.startswith('gst:'):
                    # GStreamer pipeline mode (like Mission Planner)
                    # Strip "gst:" prefix, append appsink if not present
                    pipeline = url[4:].strip()
                    if 'appsink' not in pipeline:
                        pipeline += ' ! videoconvert ! appsink drop=true max-buffers=1'
                    print(f"Video: GStreamer pipeline: {pipeline}")
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    if not cap.isOpened():
                        print("Video: GStreamer not available in cv2, falling back to FFmpeg")
                        # Fallback: try to extract rtsp URL from pipeline
                        import re
                        m = re.search(r'location=(\S+)', pipeline)
                        url = m.group(1) if m else url[4:]
                        cap = None
                if cap is None:
                    # FFmpeg mode — low latency flags like Mission Planner
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                        'rtsp_transport;udp|'
                        'fflags;nobuffer|'
                        'flags;low_delay|'
                        'framedrop;1|'
                        'probesize;32|'
                        'analyzeduration;0|'
                        'max_delay;0'
                    )
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
                        cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    print(f"Video: cannot open {RTSP_URL}")
                    video_active = False
                    socketio.emit('video_status', {'connected': False, 'error': 'Cannot open stream'})
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                print("Video: connected")
                socketio.emit('video_status', {'connected': True, 'error': None, 'w': frame_w, 'h': frame_h})
                consecutive_fails = 0
                while video_running and video_active:
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_fails += 1
                        if consecutive_fails > 10:
                            print("Video: stream lost, reconnecting...")
                            break
                        time.sleep(0.005)
                        continue
                    consecutive_fails = 0
                    h, w = frame.shape[:2]
                    if w != frame_w or h != frame_h:
                        frame_w, frame_h = w, h
                        socketio.emit('video_dims', {'w': w, 'h': h})

                    # Save raw frame for tracker thread (non-blocking)
                    with video_lock:
                        raw_frame = frame  # no copy — tracker thread will copy if needed
                        raw_frame_event.set()

                    # Encode JPEG — tracker runs in its own thread, doesn't block here
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    with video_lock:
                        video_frame = jpeg.tobytes()

                socketio.emit('video_status', {'connected': False, 'error': 'Stream lost — reconnecting...'})
        except Exception as e:
            print(f"Video capture error: {e}")
            socketio.emit('video_status', {'connected': False, 'error': str(e)})
            # Don't set video_active=False — keep trying if user wanted video
        finally:
            if cap:
                cap.release()
            with video_lock:
                video_frame = None
                raw_frame = None
            raw_frame_event.clear()
        if video_active:
            print("Video: reconnecting in 2s...")
            time.sleep(2)
        else:
            time.sleep(0.5)


def gen_frames():
    """Push frames as fast as they arrive — no artificial sleep."""
    last_sent = None
    while True:
        with video_lock:
            frame = video_frame
        if frame and frame is not last_sent:
            last_sent = frame
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.005)  # 5ms poll — don't burn CPU but stay fast


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

@app.route('/api/video/dims', methods=['GET'])
def api_video_dims():
    return jsonify({'w': frame_w, 'h': frame_h})


@app.route('/api/tracker/start', methods=['POST'])
def api_tracker_start():
    global tracker, tracker_active, tracker_bbox
    data = request.get_json() or {}
    # bbox in normalized coords 0..1
    nx = float(data.get('x', 0))
    ny = float(data.get('y', 0))
    nw = float(data.get('w', 0.1))
    nh = float(data.get('h', 0.1))

    # Wait up to 2 seconds for raw_frame if it's None (fix race condition)
    with video_lock:
        frame = raw_frame
    if frame is None:
        got_frame = raw_frame_event.wait(timeout=2.0)
        if not got_frame:
            return jsonify({'error': 'No video frame available — video not started?'}), 400
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
        t = None
        # Try NanoTrack first (best for small fast objects)
        try:
            params = cv2.TrackerNano_Params()
            params.backbone = os.path.join(BASE_DIR, 'backbone.onnx')
            params.neckhead = os.path.join(BASE_DIR, 'neckhead.onnx')
            # Also check next to exe (Windows)
            if not os.path.exists(params.backbone):
                exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else BASE_DIR
                params.backbone = os.path.join(exe_dir, 'backbone.onnx')
                params.neckhead = os.path.join(exe_dir, 'neckhead.onnx')
            if os.path.exists(params.backbone):
                t = cv2.TrackerNano_create(params)
                t.init(frame, (x, y, bw, bh))
                print("Using NanoTrack")
        except Exception as e:
            print(f"NanoTrack failed: {e}")
            t = None

        # Fallback to OpticalFlow
        if t is None:
            t = OpticalFlowTracker(frame, (x, y, bw, bh))
            print("Using OpticalFlow tracker")

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


@app.route('/api/tracker/detect', methods=['POST'])
def api_tracker_detect():
    """Return current YOLO detections immediately."""
    with yolo_detections_lock:
        dets = list(yolo_detections)
    return jsonify({'detections': dets})


@app.route('/api/tracker/detect/toggle', methods=['POST'])
def api_tracker_detect_toggle():
    """Toggle auto-detection on/off."""
    global yolo_detect_enabled
    data = request.get_json() or {}
    if 'enabled' in data:
        yolo_detect_enabled = bool(data['enabled'])
    else:
        yolo_detect_enabled = not yolo_detect_enabled
    return jsonify({'ok': True, 'enabled': yolo_detect_enabled})


@app.route('/api/guide/start', methods=['POST'])
def api_guide_start():
    """Enable GUIDE mode — start ArduPilot terminal guidance."""
    global guide_active, guide_thread, CAM_FOV_H, CAM_FOV_V
    data = request.get_json() or {}
    if 'fov_h' in data:
        CAM_FOV_H = float(data['fov_h'])
    if 'fov_v' in data:
        CAM_FOV_V = float(data['fov_v'])

    if not tracker_active:
        return jsonify({'error': 'Tracker not active — start tracker first'}), 400

    if guide_active:
        return jsonify({'ok': True, 'active': True, 'mode': 'GUIDED'})

    guide_active = True
    guide_thread = threading.Thread(target=guidance_loop, daemon=True)
    guide_thread.start()
    socketio.emit('guide_status', {'active': True, 'mode': 'GUIDED'})
    return jsonify({'ok': True, 'active': True, 'mode': 'GUIDED'})


@app.route('/api/guide/stop', methods=['POST'])
def api_guide_stop():
    """Disable GUIDE mode."""
    global guide_active
    guide_active = False
    socketio.emit('guide_status', {'active': False, 'mode': 'OFF'})
    return jsonify({'ok': True, 'active': False, 'mode': 'OFF'})


@app.route('/api/guide/status', methods=['GET'])
def api_guide_status():
    """Return current GUIDE mode status."""
    return jsonify({'active': guide_active, 'mode': 'GUIDED' if guide_active else 'OFF'})


if __name__ == '__main__':
    # Start video capture thread (waits for manual connect)
    if CV2_AVAILABLE:
        threading.Thread(target=video_capture_loop, daemon=True).start()
        threading.Thread(target=tracker_loop, daemon=True).start()
    else:
        print("cv2 not available - video disabled")

    # Start YOLO detection thread
    if YOLO_AVAILABLE:
        threading.Thread(target=yolo_detection_loop, daemon=True).start()
    else:
        print("YOLO not available - detection disabled")

    # MAVLink: manual connect only — do NOT auto-connect on startup
    if not MAVLINK_AVAILABLE:
        print("MAVLink not available - running without telemetry")

    # GCS heartbeat — needed so drone accepts commands
    threading.Thread(target=gcs_heartbeat_loop, daemon=True).start()

    telem_thread = threading.Thread(target=emit_telemetry_loop, daemon=True)
    telem_thread.start()

    print("Starting FPV app on port 8086...")
    import webbrowser
    threading.Timer(2.0, lambda: webbrowser.open('http://localhost:8086')).start()
    socketio.run(app, host='0.0.0.0', port=8086, debug=False, allow_unsafe_werkzeug=True)
