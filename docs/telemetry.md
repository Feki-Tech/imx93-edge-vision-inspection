# Telemetry & live dashboard

Per-frame results and defect alerts are published over **MQTT** — the standard
transport for industrial IoT — and visualized in a zero-backend web dashboard.

```
 i.MX93 app ──MQTT──► broker (mosquitto) ──MQTT-over-WebSockets──► dashboard/index.html
```

## 1. Start a broker (with WebSockets)

`mosquitto.conf`:

```
listener 1883
protocol mqtt

listener 9001
protocol websockets

allow_anonymous true   # demo only — use auth/TLS in production
```

```sh
mosquitto -c mosquitto.conf
# or: docker run -p 1883:1883 -p 9001:9001 -v $PWD/mosquitto.conf:/mosquitto/config/mosquitto.conf eclipse-mosquitto
```

## 2. Run the app with telemetry

```sh
pip install paho-mqtt
python -m app.main --backend cpu --source demo --mqtt-host localhost --station station1
```

Topics:

| Topic | When | Payload |
|---|---|---|
| `inspection/<station>/result` | every frame | `{ts, frame, label, score, latency_ms}` |
| `inspection/<station>/alert` | debounced defect alert (QoS 1) | same |

## 3. Open the dashboard

Open `dashboard/index.html` in a browser (a file:// open works). Query parameters:

```
dashboard/index.html?broker=ws://<broker-ip>:9001&station=station1
```

Shows live state (good/defect), frame count, alert count, latency, confidence,
and a scrolling event log. No build step, no server — MQTT.js over WebSockets.

## Production notes

- Enable broker authentication + TLS (`listener 8883`, `cafile/certfile/keyfile`).
- Set a unique `--station` per line/camera; the dashboard filters by station.
- The `alert` topic uses QoS 1 so alerts survive brief network drops.
