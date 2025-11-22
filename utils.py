from dataclasses import dataclass
from typing import List
import numpy as np
import asyncio
import json
import ssl
from pathlib import Path
from typing import AsyncGenerator, Dict, Any
import websockets
import asyncio
import time
from dataclasses import dataclass
from typing import List
import pretty_midi

HUB_IP = "stream2.mindfulmakers.xyz"   # or stream1
OUTPUT_FILE = "eeg_stream_live.jsonl"


async def eeg_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator that yields EEG dicts from the websocket.
    Also appends them to a JSONL file for logging.
    """
    print(f"Connecting to {HUB_IP}")
    print(f"Logging incoming EEG messages to {OUTPUT_FILE}")

    Path(OUTPUT_FILE).touch(exist_ok=True)

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        async with websockets.connect(f"wss://{HUB_IP}", ssl=ssl_context) as ws:
            async for msg in ws:
                try:
                    eeg = json.loads(msg)
                except json.JSONDecodeError:
                    continue  # skip malformed
                # log to disk
                f.write(json.dumps(eeg) + "\n")
                f.flush()
                # yield to whoever is consuming the stream
                yield eeg



async def run_realtime_music(style_name: str = "lofi", max_seconds: int = 30):
    """
    Stream EEG and generate music in the chosen style.
    style_name: "ambient", "lofi", ...
    """
    if style_name not in STYLE_MAP:
        raise ValueError(f"Unknown style '{style_name}'. Options: {list(STYLE_MAP.keys())}")

    style_cls = STYLE_MAP[style_name]
    style = style_cls()   # e.g. LoFiStyle() or AmbientStyle()

    start = time.time()

    async for eeg_row in eeg_stream():
        if time.time() - start > max_seconds:
            print(f"\n⏹️  Stopped streaming after {max_seconds} seconds.\n")
            break

        events = style.step(eeg_row, dt_s=0.5)
        send_to_synth(events)

    # save whatever we collected
    pm.write(f"eeg_{style_name}_live.mid")
    print(f"Saved MIDI as eeg_{style_name}_live.mid")


@dataclass
class NoteEvent:
    pitch: int
    velocity: int
    start: float
    end: float
    instrument: str        # "lead", "bass", "pad", "drums", ...
    is_drum: bool = False


class BaseStyle:
    """Base class for EEG→music styles."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.time = 0.0  # global song time in seconds

    def step(self, eeg_row, dt_s: float) -> List[NoteEvent]:
        """
        Called once per EEG chunk.
        eeg_row: dict or pandas Series with normalized EEG features.
        dt_s: how many seconds this EEG chunk covers.
        Returns: list of NoteEvent objects.
        """
        raise NotImplementedError

def band_means(row):
    """Convenience: compute hemisphere-averaged bands."""
    delta = 0.5 * (row["Left__delta"] + row["Right__delta"])
    theta = 0.5 * (row["Left__theta"] + row["Right__theta"])
    alpha = 0.5 * (row["Left__alpha"] + row["Right__alpha"])
    beta  = 0.5 * (row["Left__beta"]  + row["Right__beta"])
    gamma = 0.5 * (row["Left__gamma"] + row["Right__gamma"])
    return delta, theta, alpha, beta, gamma


def alpha_asym(row):
    return float(row["Left__alpha"] - row["Right__alpha"])  # -1..1-ish


class LoFiStyle(BaseStyle):
    """Lo-fi hip-hop style: mellow keys, soft bass, laid-back drums."""

    def step(self, row, dt_s: float) -> List[NoteEvent]:
        events: List[NoteEvent] = []

        delta, theta, alpha, beta, gamma = band_means(row)
        asym = alpha_asym(row)

        # Core parameters
        tempo = np.clip(70 + 40 * beta, 68, 92)   # BPM
        sec_per_beat = 60.0 / tempo

        # how many 8th-note slots fit in this dt_s:
        slots = max(1, int(round(dt_s / (0.5 * sec_per_beat))))

        # drives
        bass_drive = np.clip(0.6 * theta + 0.4 * delta, 0.0, 1.0)
        drum_drive = np.clip(0.5 * beta + 0.5 * gamma, 0.0, 1.0)
        energy = np.clip(max(beta, gamma), 0.1, 1.0)

        pitch_center = 60 + int(5 * asym)  # subtle left/right effect

        scale = [0, 2, 3, 5, 7, 9, 10]     # lo-fi minor-ish
        swing = 0.03                       # 30ms

        for i in range(slots):
            base_t = self.time + i * 0.5 * sec_per_beat
            if i % 2 == 1:  # swing on off-beats
                base_t += swing

            # ------ Lead ------
            if self.rng.random() < 0.5 * energy:
                deg = self.rng.integers(0, len(scale))
                pitch = pitch_center + scale[deg]
                pitch = int(np.clip(pitch, 52, 76))
                vel = int(50 + energy * 40)
                dur_beats = 1.0 if self.rng.random() < 0.7 else 0.5
                dur = dur_beats * sec_per_beat
                events.append(NoteEvent(pitch, vel, base_t, base_t + dur, "lead"))

            # ------ Bass ------
            if i % 4 == 0 and self.rng.random() < (0.3 + 0.5 * bass_drive):
                bass_root = 43  # G2-ish
                bass_pitch = bass_root + self.rng.choice([0, 7])
                bass_pitch = int(np.clip(bass_pitch, 36, 52))
                bass_vel = int(40 + bass_drive * 30)
                bass_dur = 2.0 * sec_per_beat
                events.append(NoteEvent(bass_pitch, bass_vel, base_t, base_t + bass_dur, "bass"))

            # ------ Drums ------
            KICK, SNARE, HAT = 36, 38, 42
            pos = i % 8

            # kick on 1 & 3
            if pos in (0, 4) and self.rng.random() < (0.4 + 0.4 * drum_drive):
                events.append(NoteEvent(KICK, int(40 + 30 * drum_drive),
                                        base_t, base_t + 0.25 * sec_per_beat, "drums", True))
            # snare on 2 & 4
            if pos in (2, 6) and self.rng.random() < (0.35 + 0.4 * drum_drive):
                events.append(NoteEvent(SNARE, int(40 + 30 * drum_drive),
                                        base_t, base_t + 0.25 * sec_per_beat, "drums", True))
            # hats on most 8ths
            if self.rng.random() < (0.5 + 0.4 * drum_drive):
                events.append(NoteEvent(HAT, int(30 + 25 * drum_drive),
                                        base_t, base_t + 0.15 * sec_per_beat, "drums", True))

        # advance global time
        self.time += slots * 0.5 * sec_per_beat
        return events


class AmbientStyle(BaseStyle):
    """Slow evolving pads + sparse bass, almost no drums."""

    def step(self, row, dt_s: float) -> List[NoteEvent]:
        events: List[NoteEvent] = []

        delta, theta, alpha, beta, gamma = band_means(row)
        asym = alpha_asym(row)

        tempo = 50 + 20 * theta     # slow
        sec_per_beat = 60.0 / tempo

        # we treat one EEG chunk as *one long pad*
        pad_len_beats = 4 + int(4 * theta)  # 4–8 beats
        pad_dur = pad_len_beats * sec_per_beat

        # choose a chord based on alpha asymmetry
        root = 60 + int(4 * asym)
        chord_intervals = [0, 5, 9]  # open 1–4–6 type chord

        energy = np.clip(alpha + theta, 0.1, 1.0)
        vel = int(40 + 40 * energy)

        start_t = self.time
        for iv in chord_intervals:
            pitch = int(np.clip(root + iv, 48, 84))
            events.append(NoteEvent(pitch, vel, start_t, start_t + pad_dur, "pad"))

        # simple slow bass on root
        if self.rng.random() < 0.7:
            bass_pitch = int(np.clip(root - 12, 36, 52))
            events.append(NoteEvent(bass_pitch, int(35 + 30 * delta),
                                    start_t, start_t + pad_dur, "bass"))

        # no drums or super rare soft hit could be added later

        self.time += dt_s   # time just moves with real-time here
        return events





class TechnoStyle(BaseStyle):
    """4-on-the-floor kick, driving bass, simple synth lead."""

    def step(self, row, dt_s: float) -> List[NoteEvent]:
        events: List[NoteEvent] = []

        _, _, alpha, beta, gamma = band_means(row)
        drive = np.clip(0.6 * beta + 0.4 * gamma, 0.0, 1.0)

        tempo = 120 + 40 * drive
        sec_per_beat = 60.0 / tempo

        slots = max(1, int(round(dt_s / (0.25 * sec_per_beat))))  # 16th grid
        pitch_center = 64  # D4-ish
        scale = [0, 2, 3, 7, 10]

        KICK = 36
        HAT  = 42

        for i in range(slots):
            t = self.time + i * 0.25 * sec_per_beat

            # Kick every beat (4 on floor)
            if i % 4 == 0:
                events.append(NoteEvent(KICK, int(60 + 30 * drive),
                                        t, t + 0.2 * sec_per_beat, "drums", True))

            # Hats on offbeats
            if i % 2 == 1 and self.rng.random() < (0.6 + 0.3 * drive):
                events.append(NoteEvent(HAT, int(40 + 30 * drive),
                                        t, t + 0.15 * sec_per_beat, "drums", True))

            # Bass: simple octave riff
            if i % 4 == 0 and self.rng.random() < (0.3 + 0.5 * drive):
                bass_pitch = 40 + self.rng.choice([0, 12])  # root or octave
                events.append(NoteEvent(bass_pitch, int(50 + 30 * drive),
                                        t, t + 0.5 * sec_per_beat, "bass"))

            # Lead arpeggio
            if self.rng.random() < (0.2 + 0.4 * drive):
                deg = self.rng.integers(0, len(scale))
                pitch = pitch_center + scale[deg]
                events.append(NoteEvent(pitch, int(50 + 30 * drive),
                                        t, t + 0.25 * sec_per_beat, "lead"))

        self.time += slots * 0.25 * sec_per_beat
        return events



class RockStyle(BaseStyle):
    """Straight rock-ish groove: backbeat drums, root bass, simple lead."""

    def step(self, row, dt_s: float) -> List[NoteEvent]:
        events: List[NoteEvent] = []

        delta, theta, alpha, beta, gamma = band_means(row)
        energy = np.clip(beta + gamma, 0.0, 1.0)

        tempo = 90 + 40 * energy
        sec_per_beat = 60.0 / tempo

        slots = max(1, int(round(dt_s / (0.5 * sec_per_beat))))  # 8th grid

        # root around E minor
        root = 52  # E3
        scale = [0, 2, 3, 5, 7, 10]

        KICK, SNARE, HAT = 36, 38, 42

        for i in range(slots):
            t = self.time + i * 0.5 * sec_per_beat
            pos = i % 4  # within bar (4 beats)

            # Drums: kick on 1 & 3, snare on 2 & 4
            if pos in (0, 2) and self.rng.random() < (0.7 * energy + 0.2):
                events.append(NoteEvent(KICK, int(50 + 35 * energy),
                                        t, t + 0.25 * sec_per_beat, "drums", True))
            if pos == 1 and self.rng.random() < (0.8 * energy + 0.2):
                events.append(NoteEvent(SNARE, int(55 + 35 * energy),
                                        t, t + 0.25 * sec_per_beat, "drums", True))

            # Hats every 8th
            if self.rng.random() < (0.6 + 0.3 * energy):
                events.append(NoteEvent(HAT, int(40 + 30 * energy),
                                        t, t + 0.2 * sec_per_beat, "drums", True))

            # Bass: root notes on beats
            if pos == 0 and self.rng.random() < (0.7 + 0.2 * energy):
                bass_pitch = int(np.clip(root, 40, 55))
                events.append(NoteEvent(bass_pitch, int(45 + 35 * energy),
                                        t, t + 1.0 * sec_per_beat, "bass"))

            # Lead: short pentatonic licks
            if self.rng.random() < (0.3 + 0.4 * energy):
                deg = self.rng.integers(0, len(scale))
                pitch = root + 12 + scale[deg]  # one octave up
                events.append(NoteEvent(pitch, int(50 + 40 * energy),
                                        t, t + 0.5 * sec_per_beat, "lead"))

        self.time += slots * 0.5 * sec_per_beat
        return events
    

    # Global MIDI container
pm = pretty_midi.PrettyMIDI(initial_tempo=80)
lead_inst = pretty_midi.Instrument(program=4)   # Electric piano
bass_inst = pretty_midi.Instrument(program=34)  # Bass
drum_inst = pretty_midi.Instrument(program=0)
drum_inst.is_drum = True

pm.instruments.extend([lead_inst, bass_inst, drum_inst])

def send_to_synth(events: List[NoteEvent]):
    """For now, 'synth' = build up a MIDI file in memory."""
    for ev in events:
        note = pretty_midi.Note(
            pitch=ev.pitch,
            velocity=ev.velocity,
            start=ev.start,
            end=ev.end,
        )
        if ev.is_drum:
            drum_inst.notes.append(note)
        elif ev.instrument == "bass":
            bass_inst.notes.append(note)
        else:  # default to lead
            lead_inst.notes.append(note)


# at top of utils.py (after AmbientStyle, LoFiStyle are defined)
STYLE_MAP = {
    "ambient": AmbientStyle,
    "lofi": LoFiStyle,
    "rock": RockStyle,
    "techno": TechnoStyle,
    # later: "techno": TechnoStyle
}
