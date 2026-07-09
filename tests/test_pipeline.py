import numpy as np

from app.decision import DefectDecider
from app.frame_source import demo_frames
from app.inference import HeuristicEngine, InferenceResult, make_engine


def test_demo_frames_shape():
    frames = list(demo_frames(count=8))
    assert len(frames) == 8
    assert all(f.shape == (224, 224, 3) and f.dtype == np.uint8 for f in frames)


def test_heuristic_detects_injected_defects():
    engine = HeuristicEngine()
    results = [engine.infer(f) for f in demo_frames(count=20)]
    labels = [r.label for r in results]
    assert "defect" in labels and "good" in labels


def test_make_engine_falls_back_without_model():
    assert make_engine("cpu", model_path="does/not/exist.tflite").name == "heuristic"


def test_decider_debounce():
    d = DefectDecider(threshold=0.5, debounce=2)
    hit = InferenceResult("defect", 0.9, 1.0)
    miss = InferenceResult("good", 0.9, 1.0)
    assert d.update(hit) is False       # first hit: not yet
    assert d.update(hit) is True        # second consecutive hit: alert
    assert d.update(miss) is False      # reset by good frame
