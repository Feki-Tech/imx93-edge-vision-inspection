"""MQTT telemetry: publishes per-frame results and defect alerts as JSON.

Optional dependency (paho-mqtt). If no broker is configured, telemetry is a
no-op so the pipeline runs unchanged.

Topics:
    inspection/<station>/result   every frame  {ts, frame, label, score, latency_ms}
    inspection/<station>/alert    on debounced defect alert
"""

from __future__ import annotations

import json
import time

from .inference import InferenceResult


class Telemetry:
    def __init__(self, host: str | None, port: int = 1883, station: str = "station1"):
        self.client = None
        self.station = station
        if not host:
            return
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.connect(host, port)
        self.client.loop_start()

    def publish(self, frame_idx: int, result: InferenceResult, alert: bool) -> None:
        if not self.client:
            return
        payload = {
            "ts": time.time(),
            "frame": frame_idx,
            "label": result.label,
            "score": round(result.score, 4),
            "latency_ms": round(result.latency_ms, 2),
        }
        base = f"inspection/{self.station}"
        self.client.publish(f"{base}/result", json.dumps(payload))
        if alert:
            self.client.publish(f"{base}/alert", json.dumps(payload), qos=1)

    def close(self) -> None:
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
