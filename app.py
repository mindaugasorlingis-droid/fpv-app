#!/usr/bin/env python3
"""
FPV Ground Station - MAVLink WebSocket proxy + Flask backend
"""

import threading
import time
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# MAVLink import
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False
    print("WARNING: pymavlink not available")

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'fpv-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global MAVLink connection and state
mav_connection = None
mav_lock = threading.Lock()
mav_connect_string = 'udpin:0.0.0.0:14550'
mav_status = {'connected': False, 'connecting': False, 'error': None, 'connection_string': 'udpin:0.0.0.0:14550'}
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


def connect_mavlink(connection_string=None):
    """Connect to MAVLink and keep reconnecting."""
    global mav_connection, mav_connect_string, mav_status, mav_stop_event
    if connection_string:
        mav_connect_string = connection_string
    mav_stop_event.clear()
    while not mav_stop_event.is_set():
        try:
            cs = mav_connect_string
            print(f"Connecting to MAVLink on {cs}...")
            mav_status['connecting'] = True
            mav_status['connected'] = False
            mav_status['error'] = None
            mav_status['connection_string'] = cs
            conn = mavutil.mavlink_connection(cs, source_system=255, source_component=0)
            conn.wait_heartbeat(timeout=30)
            print(f"MAVLink connected: system {conn.target_system}, component {conn.target_component}")
            with mav_lock:
                mav_connection = conn
            mav_status['connected'] = True
            mav_status['connecting'] = False
            mav_status['error'] = None
            receive_loop(conn)
        except Exception as e:
            print(f"MAVLink connection error: {e}")
            mav_status['connected'] = False
            mav_status['connecting'] = False
            mav_status['error'] = str(e)
            with mav_lock:
                mav_connection = None
            if not mav_stop_event.is_set():
                time.sleep(5)


def receive_loop(conn):
    """Receive and process MAVLink messages."""
    global telemetry
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


@app.route('/api/connection', methods=['GET'])
def api_connection_status():
    return jsonify(mav_status)


@app.route('/api/connect', methods=['POST'])
def api_connect():
    global mav_connect_string, mav_stop_event
    data = request.get_json() or {}
    ip = data.get('ip', '').strip()
    port = data.get('port', '14550')
    conn_type = data.get('type', 'udpci')  # udpci, udpin, tcp

    if not ip:
        return jsonify({'error': 'IP is required'}), 400

    # Build connection string
    baud = int(data.get('baud', 115200))
    if conn_type == 'udpci':
        cs = f'udpci:{ip}:{port}'
    elif conn_type == 'udpin':
        cs = f'udpin:0.0.0.0:{port}'
    elif conn_type == 'tcp':
        cs = f'tcp:{ip}:{port}'
    elif conn_type == 'serial':
        cs = f'{port},{baud}'  # pymavlink serial format
    else:
        cs = f'udpci:{ip}:{port}'

    # Stop current connection thread
    mav_stop_event.set()
    with mav_lock:
        pass  # let receive_loop exit naturally

    time.sleep(0.5)

    # Start new connection thread
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

if __name__ == '__main__':
    if MAVLINK_AVAILABLE:
        mav_thread = threading.Thread(target=connect_mavlink, daemon=True)
        mav_thread.start()
    else:
        print("MAVLink not available - running without telemetry")

    telem_thread = threading.Thread(target=emit_telemetry_loop, daemon=True)
    telem_thread.start()

    print("Starting FPV app on port 8086...")
    socketio.run(app, host='0.0.0.0', port=8086, debug=False)
