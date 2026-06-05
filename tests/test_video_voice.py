from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from familyrobot.video_voice import (
    VideoAudioExtractor,
    VideoMetadata,
    VideoWakeEventFinder,
    WavAudioPlayer,
    build_recorded_video_alignment,
    read_video_metadata,
)
from familyrobot.voice import VoiceTranscriptSegment, VoskTranscriber


class _FakeCapture:
    def __init__(self, fps: float, frame_count: int) -> None:
        self.fps = fps
        self.frame_count = frame_count
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def get(self, prop: int) -> float:
        if prop == 5:
            return self.fps
        if prop == 7:
            return self.frame_count
        return 0.0

    def release(self) -> None:
        self.released = True


class _SegmentRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802
        self.calls += 1
        return self.calls == 1

    def Result(self) -> str:  # noqa: N802
        return '{"text": "你好 爸爸", "result": [{"start": 0.25, "end": 0.70}]}'

    def FinalResult(self) -> str:  # noqa: N802
        return '{"text": "你好 爸爸", "result": [{"start": 0.25, "end": 0.70}]}'


def test_read_video_metadata_reads_fps_and_frame_count(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    fake_capture = _FakeCapture(fps=25.0, frame_count=250)
    fake_cv2 = SimpleNamespace(VideoCapture=lambda path: fake_capture, CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    metadata = read_video_metadata(video_path)

    assert metadata == VideoMetadata(video_path=video_path, fps=25.0, frame_count=250)
    assert metadata.timestamp_for_frame(25) == 1.0
    assert metadata.frame_for_timestamp(2.0) == 50
    assert metadata.duration_seconds == 10.0
    assert fake_capture.released is True


def test_video_audio_extractor_builds_ffmpeg_command(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    commands: list[list[str]] = []

    def fake_run(command, check, stdout, stderr):  # noqa: ANN001
        commands.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("familyrobot.video_voice.subprocess.run", fake_run)

    extractor = VideoAudioExtractor(ffmpeg_executable="ffmpeg", output_dir=tmp_path / "audio")
    audio_path = extractor.extract(video_path)

    assert audio_path == tmp_path / "audio" / "demo.wav"
    assert commands[0][0] == "ffmpeg"
    assert "-vn" in commands[0]
    assert "-ar" in commands[0]
    assert str(audio_path) in commands[0]


def test_build_recorded_video_alignment_uses_audio_extractor(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "demo.wav"
    calls: list[Path] = []

    def fake_read_video_metadata(path: str | Path) -> VideoMetadata:
        return VideoMetadata(video_path=Path(path), fps=30.0, frame_count=300)

    class _FakeExtractor:
        def extract(self, path: str | Path, output_dir: str | Path | None = None) -> Path:
            calls.append(Path(path))
            return audio_path

    monkeypatch.setattr("familyrobot.video_voice.read_video_metadata", fake_read_video_metadata)

    alignment = build_recorded_video_alignment(video_path, extractor=_FakeExtractor())

    assert calls == [video_path]
    assert alignment.audio_path == audio_path
    assert alignment.frame_for_timestamp(1.0) == 30
    assert alignment.timestamp_for_frame(60) == 2.0


def test_vosk_transcriber_returns_timestamped_segments(monkeypatch, tmp_path: Path) -> None:
    wav_path = tmp_path / "voice.wav"
    wav_path.write_bytes(b"wave")

    monkeypatch.setattr("familyrobot.voice.wavfile.read", lambda path: (16000, np.ones(3200, dtype=np.int16)))

    transcriber = VoskTranscriber(
        model_loader=lambda model_path: object(),
        recognizer_factory=lambda model, sample_rate: _SegmentRecognizer(),
    )

    segments = transcriber.transcribe_wav_segments(wav_path, chunk_seconds=0.1)

    assert segments
    assert segments[0].text == "你好 爸爸"
    assert segments[0].start_seconds == 0.25
    assert segments[0].end_seconds == 0.70


def test_video_wake_event_finder_maps_wake_segments_to_frames() -> None:
    alignment = type(
        "Alignment",
        (),
        {
            "frame_for_timestamp": staticmethod(lambda timestamp: int(timestamp * 30)),
        },
    )()
    segment = VoiceTranscriptSegment(
        text="你好 爸爸",
        raw_json='{"text": "你好 爸爸"}',
        start_seconds=1.2,
        end_seconds=1.8,
    )

    class _FakeTranscriber:
        def transcribe_wav_segments(self, audio_path):  # noqa: ANN001
            return [segment]

    finder = VideoWakeEventFinder(transcriber=_FakeTranscriber(), wake_phrase="你好")
    events = finder.find_events(Path("dummy.wav"), alignment)

    assert len(events) == 1
    assert events[0].frame_index == 36
    assert events[0].timestamp_seconds == 1.2


def test_wav_audio_player_uses_sounddevice(monkeypatch, tmp_path: Path) -> None:
    wav_path = tmp_path / "audio.wav"
    from scipy.io import wavfile

    wavfile.write(wav_path, 16000, np.array([0, 1000, -1000, 0], dtype=np.int16))

    play_calls: list[tuple[np.ndarray, int]] = []

    fake_sd = SimpleNamespace(
        play=lambda data, rate: play_calls.append((np.asarray(data), rate)),
        wait=lambda: None,
        stop=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    player = WavAudioPlayer()
    player.start(wav_path)
    player.stop()

    assert play_calls
    assert play_calls[0][1] == 16000
    assert play_calls[0][0].shape[0] == 4
