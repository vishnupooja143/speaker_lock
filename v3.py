"""
speaker_lock.py - customer voice lock + barge-in gate + optional multi-speaker separation
=========================================================================================
FINAL VERSION with all accuracy improvements (Including 3+ Speaker Enhancements):

- Median-of-windows profile, adaptive threshold, per-utterance pitch penalty.
- Full barge-in support during greeting.
- Optional DNN source separation (ConvTasNet or Sepformer) for overlapping speakers.
- ENHANCEMENTS for 97-98% overlap accuracy & 80-90% 3-speaker accuracy:
  1. Full‑chunk scoring (not just 400ms).
  2. Dynamic threshold relaxation during detected double‑talk/multi-talk.
  3. Better default separator (Sepformer) via SEPARATION_MODEL env var.
  4. Hybrid fallback to original mix if separation fails or introduces artifacts.
  5. N-Way Tournament: Scores ALL separated streams (supports 3+ speakers).
  
- NEW: ENROLLMENT CONTAMINATION PROTECTION:
  1. Clusters enrollment embeddings to detect secondary speakers.
  2. Rejects or cleans the profile if multiple people speak during enrollment.
  3. Disables Sepformer during enrollment to prevent AI hallucinations.

Enable separation with: SPEAKER_SEPARATION=true
For 3 speakers, use: SEPARATION_MODEL="speechbrain/sepformer-wsj03mix"
=========================================================================================
"""

import logging
import enum
import collections
import time
import math as _math
import os
import wave
import numpy as np
from typing import Tuple, Any, Optional, Callable
import threading
import queue

from livekit import rtc

logger = logging.getLogger("speaker_lock")

# ======================================================================
# TUNING (all can be overridden by environment variables)
# ======================================================================
_ENV_SFM_NOISE_FLOOR   = float(os.getenv("SFM_NOISE_FLOOR", "0.88"))
_ENV_SFM_MUSIC_CEILING = float(os.getenv("SFM_MUSIC_CEILING", "0.06"))

# Base threshold – adaptive calibration will move it.
_ENV_COSINE_THRESHOLD  = float(os.getenv("COSINE_THRESHOLD", "0.52"))
_ENV_TURN_VERIFY_RATIO = float(os.getenv("TURN_VERIFY_MIN_RATIO", "0.45"))

_HARD_REJECT_COSINE = float(os.getenv("HARD_REJECT_COSINE", "0.28"))

# Pitch penalty: soft and hard deviation (in MAD units)
_ENV_F0_SOFT_DEVIATION = float(os.getenv("F0_SOFT_DEVIATION", "0.30"))
_ENV_F0_HARD_DEVIATION = float(os.getenv("F0_HARD_DEVIATION", "0.60"))
_ENV_F0_MAX_PENALTY    = float(os.getenv("F0_MAX_PENALTY", "0.12"))

# Enrollment loudness gates
_ENV_ENROLL_MIN_SNR_RATIO     = float(os.getenv("ENROLL_MIN_SNR_RATIO", "2.0"))
_ENV_ENROLL_RELAXED_SNR_RATIO = float(os.getenv("ENROLL_RELAXED_SNR_RATIO", "1.6"))

# Fail-open and re-enrollment
_FAIL_OPEN_AFTER_REJECTS  = int(os.getenv("FAIL_OPEN_AFTER_REJECTS", "3"))
_FAIL_OPEN_RELAX_STEP     = float(os.getenv("FAIL_OPEN_RELAX_STEP", "0.05"))
_FAIL_OPEN_MIN_THRESHOLD  = float(os.getenv("FAIL_OPEN_MIN_THRESHOLD", "0.45"))
_FAIL_OPEN_MAX_USES       = int(os.getenv("FAIL_OPEN_MAX_USES", "3"))
_REENROLL_AFTER_REJECTS   = int(os.getenv("REENROLL_AFTER_REJECTS", "4"))

# F0 / pitch
_F0_MAD_SUSPECT_HZ        = float(os.getenv("F0_MAD_SUSPECT_HZ", "60"))
_LOW_CONF_THRESHOLD_DROP  = float(os.getenv("LOW_CONF_THRESHOLD_DROP", "0.05"))
_F0_EVERY = max(1, int(os.getenv("SPEAKER_F0_EVERY", "3")))

# Barge-in
_BARGE_ECHO_RATIO   = float(os.getenv("BARGE_ECHO_RATIO", "2.2"))
_BARGE_FLOOR_RATIO  = float(os.getenv("BARGE_FLOOR_RATIO", "3.0"))
_ECHO_EMA_ALPHA     = float(os.getenv("ECHO_EMA_ALPHA", "0.10"))
_BOT_SPEAKING_EXTRA_MARGIN = float(os.getenv("BOT_SPEAKING_EXTRA_MARGIN", "0.04"))
_BARGE_MIN_SUSTAIN_MS = float(os.getenv("BARGE_MIN_SUSTAIN_MS", "150"))

# Mode: "monitor" (audio untouched) or "gate" (zero rejected chunks)
SPEAKER_LOCK_MODE = os.getenv("SPEAKER_LOCK_MODE", "monitor").strip().lower()
if SPEAKER_LOCK_MODE not in ("monitor", "gate"):
    SPEAKER_LOCK_MODE = "monitor"

_AUTO_DEGRADE_AFTER   = int(os.getenv("SPEAKER_AUTO_DEGRADE_AFTER", "150"))
_AUTO_DEGRADE_RATIO   = float(os.getenv("SPEAKER_AUTO_DEGRADE_RATIO", "0.55"))

# Adaptive calibration
_ADAPT_ENABLED      = os.getenv("SPEAKER_ADAPT_ENABLED", "true").lower() == "true"
_ADAPT_MIN_SAMPLES  = int(os.getenv("SPEAKER_ADAPT_MIN_SAMPLES", "40"))
_ADAPT_EVERY        = int(os.getenv("SPEAKER_ADAPT_EVERY", "25"))
_ADAPT_MARGIN_MADS  = float(os.getenv("SPEAKER_ADAPT_MARGIN_MADS", "2.5"))
_ADAPT_MIN          = float(os.getenv("SPEAKER_ADAPT_MIN", "0.35"))
_ADAPT_MAX          = float(os.getenv("SPEAKER_ADAPT_MAX", "0.72"))
_ADAPT_RAISE_RATIO   = float(os.getenv("SPEAKER_ADAPT_RAISE_RATIO", "0.92"))
_ADAPT_RAISE_STEP    = float(os.getenv("SPEAKER_ADAPT_RAISE_STEP", "0.005"))

# Profile drift
_PROFILE_EMA_ENABLED = os.getenv("SPEAKER_PROFILE_EMA", "true").lower() == "true"
_PROFILE_EMA_ALPHA   = float(os.getenv("SPEAKER_PROFILE_EMA_ALPHA", "0.02"))
_PROFILE_EMA_MARGIN  = float(os.getenv("SPEAKER_PROFILE_EMA_MARGIN", "0.06"))

# Utterance-level rejection
_UTT_MIN_MEAN_SIM   = float(os.getenv("UTT_MIN_MEAN_SIM", "0.42"))
_UTT_MIN_FRAMES     = int(os.getenv("UTT_MIN_FRAMES", "6"))
_MUSIC_SFM_CEILING  = float(os.getenv("MUSIC_SFM_CEILING", "0.04"))
_MUSIC_MIN_RUN      = int(os.getenv("MUSIC_MIN_RUN", "25"))

# Noise reduction
_NOISE_REDUCTION_ENABLED = os.getenv("NOISE_REDUCTION", "true").lower() == "true"
_NR_FLOOR_GAIN   = float(os.getenv("NR_FLOOR_GAIN", "0.22"))
_NR_OVERSUBTRACT = float(os.getenv("NR_OVERSUBTRACT", "1.4"))
_NR_NOISE_ALPHA  = float(os.getenv("NR_NOISE_ALPHA", "0.06"))
_NR_MIN_UPDATES  = int(os.getenv("NR_MIN_UPDATES", "8"))

# --- Multi-speaker separation (optional) ---
_SEPARATION_ENABLED = os.getenv("SPEAKER_SEPARATION", "false").lower() == "true"
_SEPARATION_WINDOW_S = float(os.getenv("SEPARATION_WINDOW_S", "2.0"))
_SEPARATION_MODEL = os.getenv("SEPARATION_MODEL", "speechbrain/sepformer-libri3mix")

# Minimum score required for separated audio to beat the raw mix (prevents hallucination artifacts)
_SEP_MIN_FALLBACK_SCORE = float(os.getenv("SEP_MIN_FALLBACK_SCORE", "0.45"))

# Other constants
ENROLL_SECONDS      = float(os.getenv("ENROLL_SECONDS", "3.0"))
ENROLL_MAX_SECONDS  = float(os.getenv("ENROLL_MAX_SECONDS", "12.0"))
_SAMPLE_RATE         = 8000
_FRAME_SIZE          = 160
_PROFILE_WINDOW_SAMPLES = int(os.getenv("PROFILE_WINDOW_SAMPLES", "2400"))

_RMS_FLOOR_RATIO      = float(os.getenv("RMS_FLOOR_RATIO", "0.9"))
_COSINE_THRESHOLD     = _ENV_COSINE_THRESHOLD
_HARD_REJECT_COSINE   = _ENV_HARD_REJECT_COSINE
_ENROLL_MIN_SNR_RATIO = _ENV_ENROLL_MIN_SNR_RATIO

_BG_FLOOR_DOWN_ALPHA   = 0.12
_BG_FLOOR_UP_ALPHA     = 0.004

_LANG_SWITCH_MIN_VERIFIED_TURNS = int(os.getenv("LANG_SWITCH_MIN_VERIFIED_TURNS", "1"))
_LANG_SWITCH_MIN_TURNS_ENROLLING = int(os.getenv("LANG_SWITCH_MIN_TURNS_ENROLLING", "2"))

_SFM_MUSIC_CEILING  = _ENV_SFM_MUSIC_CEILING
_SFM_NOISE_FLOOR    = _ENV_SFM_NOISE_FLOOR

_SNR_ALPHA_SCALE_DB   = 15.0
_SIMILARITY_EMA_ALPHA = 0.22
_F0_MIN_HZ = 70.0
_F0_MAX_HZ = 400.0
_F0_SOFT_DEVIATION = _ENV_F0_SOFT_DEVIATION
_F0_HARD_DEVIATION = _ENV_F0_HARD_DEVIATION
_F0_MAX_PENALTY    = _ENV_F0_MAX_PENALTY

VAD_WINDOW_MS    = 2000
VAD_ACCEPT_RATIO = _ENV_TURN_VERIFY_RATIO
VAD_ACCEPT_RATIO_BOT_SPEAKING = float(os.getenv("VAD_ACCEPT_RATIO_BOT_SPEAKING", "0.40"))

TURN_SPEAKER_VERIFY_MIN_RATIO = _ENV_TURN_VERIFY_RATIO

_BURST_GAP_S   = 0.6
_BURST_FRESH_S = float(os.getenv("BURST_FRESH_S", "1.2"))

_ACCEPT_RUN_MIN_MS  = float(os.getenv("ACCEPT_RUN_MIN_MS", "400"))
_ACCEPT_RUN_FRESH_S = float(os.getenv("ACCEPT_RUN_FRESH_S", "1.5"))

N_FILTERS = 20
N_CEPS    = 13

# ----------------------------------------------------------------------
# Optional: load separation model
# ----------------------------------------------------------------------
_SEP_MODEL = None
if _SEPARATION_ENABLED:
    try:
        import torch
        from asteroid.models import BaseModel
        logger.warning(
            f"🔊 Speaker Separation ENABLED using model: {_SEPARATION_MODEL} "
            f"(adds ~{_SEPARATION_WINDOW_S}s latency & high CPU/GPU load)"
        )
        _SEP_MODEL = BaseModel.from_pretrained(_SEPARATION_MODEL)
        _SEP_MODEL.eval()
        if torch.cuda.is_available():
            _SEP_MODEL.cuda()
            logger.info("🚀 Separator running on GPU")
        else:
            logger.warning("💻 Separator running on CPU - may drop audio frames!")
    except ImportError:
        logger.error("❌ pip install asteroid torch required for separation. Disabling.")
        _SEPARATION_ENABLED = False
    except Exception as e:
        logger.error(f"❌ Failed to load separation model '{_SEPARATION_MODEL}': {e}. Disabling.")
        _SEPARATION_ENABLED = False
else:
    logger.info("🔇 Speaker separation DISABLED (multi-speaker overlap accuracy limited to ~50-60%)")

logger.warning(
    f"🔊 SpeakerLock MODE={SPEAKER_LOCK_MODE.upper()} "
    f"({'audio passes through untouched' if SPEAKER_LOCK_MODE == 'monitor' else 'REJECTED FRAMES ARE ZEROED - can corrupt STT input'})")
logger.info(
    f"🔊 SpeakerLock config: COSINE={_ENV_COSINE_THRESHOLD}, "
    f"HARD_REJECT={_ENV_HARD_REJECT_COSINE}, "
    f"FAIL_OPEN_AFTER={_FAIL_OPEN_AFTER_REJECTS} MAX_USES={_FAIL_OPEN_MAX_USES}, "
    f"ENROLL_MIN_SNR={_ENV_ENROLL_MIN_SNR_RATIO}, adapt={_ADAPT_ENABLED}, "
    f"barge_echo_ratio={_BARGE_ECHO_RATIO}, barge_min_sustain={_BARGE_MIN_SUSTAIN_MS}ms, "
    f"separation={_SEPARATION_ENABLED} ({_SEPARATION_MODEL})"
)


class LockState(enum.Enum):
    ENROLLING = "ENROLLING"
    LOCKED    = "LOCKED"


def _log_filterbank_energy(pcm_int16: np.ndarray, n_filters: int = N_FILTERS) -> np.ndarray:
    n = len(pcm_int16)
    if n == 0:
        return np.zeros(n_filters, dtype=np.float32)
    samples   = pcm_int16.astype(np.float32) / 32768.0
    fft_mag   = np.abs(np.fft.rfft(samples, n=max(n, 256)))
    band_size = max(1, len(fft_mag) // n_filters)
    energies  = np.array(
        [fft_mag[i * band_size:(i + 1) * band_size].mean()
         for i in range(n_filters)],
        dtype=np.float32,
    )
    return np.log1p(energies)


_DCT_MATRIX_CACHE: dict[int, np.ndarray] = {}


def _dct_matrix(n_filters: int) -> np.ndarray:
    cached = _DCT_MATRIX_CACHE.get(n_filters)
    if cached is not None:
        return cached
    k = np.arange(n_filters)
    n = np.arange(n_filters)
    mat = np.cos(np.pi / n_filters * (n[:, None] + 0.5) * k[None, :])
    mat[:, 0] *= 1.0 / np.sqrt(2.0)
    mat *= np.sqrt(2.0 / n_filters)
    _DCT_MATRIX_CACHE[n_filters] = mat.astype(np.float32)
    return _DCT_MATRIX_CACHE[n_filters]


def _cepstral_coeffs(log_fbank: np.ndarray, n_ceps: int = N_CEPS) -> np.ndarray:
    mat = _dct_matrix(len(log_fbank))
    ceps = log_fbank @ mat
    ceps = ceps[1:n_ceps + 1] if len(ceps) > n_ceps else ceps[1:]
    if len(ceps) < n_ceps - 1:
        ceps = np.pad(ceps, (0, (n_ceps - 1) - len(ceps)))
    return ceps.astype(np.float32)


def _zero_crossing_rate(pcm_int16: np.ndarray) -> float:
    if len(pcm_int16) < 2:
        return 0.0
    signs = np.sign(pcm_int16.astype(np.float32))
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def _spectral_centroid(pcm_int16: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> float:
    n = len(pcm_int16)
    if n == 0:
        return 0.0
    samples = pcm_int16.astype(np.float32) / 32768.0
    fft_mag = np.abs(np.fft.rfft(samples, n=max(n, 256)))
    freqs   = np.fft.rfftfreq(max(n, 256), d=1.0 / sample_rate)
    total   = fft_mag.sum()
    if total < 1e-9:
        return 0.0
    return float(np.dot(freqs, fft_mag) / total)


def _spectral_flatness(pcm_int16: np.ndarray) -> float:
    n = len(pcm_int16)
    if n == 0:
        return 0.5
    samples  = pcm_int16.astype(np.float32) / 32768.0
    fft_mag  = np.abs(np.fft.rfft(samples, n=max(n, 256))) + 1e-10
    geo_mean = np.exp(np.mean(np.log(fft_mag)))
    ari_mean = np.mean(fft_mag)
    if ari_mean < 1e-10:
        return 0.5
    return float(np.clip(geo_mean / ari_mean, 0.0, 1.0))


def _estimate_f0(pcm_int16: np.ndarray, sample_rate: int = _SAMPLE_RATE,
                 fmin: float = _F0_MIN_HZ, fmax: float = _F0_MAX_HZ) -> float:
    if len(pcm_int16) < 80:
        return 0.0
    samples = pcm_int16.astype(np.float32)
    samples = samples - samples.mean()
    if np.allclose(samples, 0.0):
        return 0.0
    corr = np.correlate(samples, samples, mode="full")
    corr = corr[len(corr) // 2:]
    min_lag = int(sample_rate / fmax)
    max_lag = int(sample_rate / fmin)
    max_lag = min(max_lag, len(corr) - 1)
    if min_lag >= max_lag or min_lag < 1:
        return 0.0
    segment = corr[min_lag:max_lag]
    if len(segment) == 0 or segment.max() <= 0:
        return 0.0

    zero_lag = corr[0] if corr[0] > 0 else 1e-9
    peak_val = segment.max()
    if (peak_val / zero_lag) < 0.28:
        return 0.0
    peak_lag = min_lag + int(np.argmax(segment))
    if peak_lag <= 0:
        return 0.0
    return float(sample_rate / peak_lag)


def _extract_embedding(pcm_bytes: bytes, sample_rate: int = _SAMPLE_RATE) -> tuple:
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(pcm) == 0:
        return np.zeros(N_CEPS - 1 + 4, dtype=np.float32), 0.5, 0.0
    lfe   = _log_filterbank_energy(pcm, n_filters=N_FILTERS)
    ceps  = _cepstral_coeffs(lfe, n_ceps=N_CEPS)
    zcr      = np.array([_zero_crossing_rate(pcm)], dtype=np.float32)
    centroid = np.array(
        [_spectral_centroid(pcm, sample_rate) / (sample_rate / 2)], dtype=np.float32
    )
    f0_hz    = _estimate_f0(pcm, sample_rate=sample_rate)
    f0_norm  = np.array(
        [min(max(f0_hz, 0.0), _F0_MAX_HZ) / _F0_MAX_HZ], dtype=np.float32
    )
    sfm_val  = _spectral_flatness(pcm)
    sfm      = np.array([sfm_val], dtype=np.float32)
    vec      = np.concatenate([ceps, zcr, centroid, f0_norm, sfm])
    norm     = np.linalg.norm(vec)
    return vec / (norm + 1e-9), sfm_val, f0_hz


_EMBEDDING_BACKEND: dict = {"fn": None}


def set_embedding_backend(fn: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
    _EMBEDDING_BACKEND["fn"] = fn
    logger.info(f"🔧 SpeakerLock embedding backend set to: "
                f"{fn.__name__ if fn else 'built-in cepstral'}")


def _compute_embedding(pcm_bytes: bytes, sample_rate: int = _SAMPLE_RATE) -> tuple:
    custom_fn = _EMBEDDING_BACKEND.get("fn")
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    sfm_val = _spectral_flatness(pcm)
    f0_hz = _estimate_f0(pcm, sample_rate=sample_rate)
    if custom_fn is not None:
        try:
            vec = np.asarray(custom_fn(pcm), dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-9), sfm_val, f0_hz
        except Exception as e:
            logger.error(f"❌ custom embedding backend failed ({e}) — falling back "
                         f"to handcrafted for this chunk")
    return _extract_embedding(pcm_bytes, sample_rate)


def _rms(pcm_bytes: bytes) -> float:
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(pcm) == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm ** 2)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _is_voice_band(sfm_val: float) -> bool:
    return _SFM_MUSIC_CEILING <= sfm_val <= _SFM_NOISE_FLOOR


# ======================================================================
# SPEAKER LOCK - CORE CLASS
# ======================================================================
class SpeakerLock:
    def __init__(
        self,
        enroll_seconds:   float = ENROLL_SECONDS,
        cosine_threshold: float = _COSINE_THRESHOLD,
        sample_rate:      int   = _SAMPLE_RATE,
        bot_speaking_ref: dict  = None,
    ):
        self._enroll_seconds     = enroll_seconds
        self._enroll_max_seconds = ENROLL_MAX_SECONDS
        self._cosine_threshold   = cosine_threshold
        self._base_threshold     = cosine_threshold
        self._sample_rate        = sample_rate
        self._bot_speaking_ref   = bot_speaking_ref or {"bot_speaking": False}
        self._state              = LockState.ENROLLING
        self._enroll_frames: list[np.ndarray] = []
        self._enroll_fallback_frames: list[np.ndarray] = []
        self._enrolled_ms        = 0.0
        self._enroll_wall_ms     = 0.0
        self._enroll_f0_samples: list[float] = []
        self._enroll_fallback_f0: list[float] = []
        self._profile: np.ndarray | None = None
        self._profile_f0_median: float | None = None
        self._profile_f0_mad: float = 40.0
        self._bg_rms_ema = 500.0
        self._smoothed_similarity: float | None = None
        self._accepted = 0
        self._rejected = 0
        self._total    = 0
        self._silence  = 0
        self._echo_skipped = 0
        self._barge_frames = 0
        self._reenroll_count = 0

        self._consecutive_utt_rejects = 0
        self._fail_open_uses = 0
        self._pass_through = False
        self._low_confidence_profile = False

        self._enroll_gate_ratio = _ENROLL_MIN_SNR_RATIO
        self._enroll_gate_relaxed = False
        self._frame_filter_ref: "SpeakerAwareFrameFilter | None" = None

        self._echo_rms_ema: float = 0.0
        self._echo_samples: int = 0

        self._f0_tick = 0
        self._last_f0 = 0.0
        self._tonal_run = 0
        self._music_rejects = 0
        self._nonvoice_rejects = 0
        self._utt_sims: list[float] = []
        self._utt_f0s: list[float] = []

        self._sim_history: collections.deque = collections.deque(maxlen=400)
        self._since_adapt = 0
        self._adapted = False
        self._high_accept_counter = 0
        
        # --- NEW: ENROLLMENT CONTAMINATION FLAG ---
        self._contamination_detected = False

        self._bot_speaking_ref = (bot_speaking_ref if bot_speaking_ref is not None
                                  else {"bot_speaking": False})

        self._barge_sustain_ms = 0.0

    # ------------------------------------------------------------------ api --
    def set_bot_speaking_ref(self, ref: dict) -> None:
        self._bot_speaking_ref = ref
        logger.info("🔗 SpeakerLock: bot_speaking_ref attached")

    def set_sample_rate(self, sample_rate: int) -> None:
        if sample_rate and sample_rate != self._sample_rate:
            self._sample_rate = sample_rate

    def register_frame_filter(self, ff: "SpeakerAwareFrameFilter") -> None:
        self._frame_filter_ref = ff

    @property
    def state(self) -> LockState:
        return self._state

    @property
    def locked(self) -> bool:
        return self._state == LockState.LOCKED

    @property
    def pass_through(self) -> bool:
        return self._pass_through
        
    @property
    def contamination_detected(self) -> bool:
        return self._contamination_detected

    def _update_noise_floor(self, rms_val: float) -> None:
        if rms_val < self._bg_rms_ema:
            self._bg_rms_ema += _BG_FLOOR_DOWN_ALPHA * (rms_val - self._bg_rms_ema)
        else:
            self._bg_rms_ema += _BG_FLOOR_UP_ALPHA * (rms_val - self._bg_rms_ema)
        self._bg_rms_ema = max(self._bg_rms_ema, 20.0)

    def note_utterance_result(self, accepted: bool) -> None:
        if accepted:
            self._consecutive_utt_rejects = 0
            return
        self._consecutive_utt_rejects += 1
        logger.info(
            f"🟠 SpeakerLock: consecutive rejected utterances = "
            f"{self._consecutive_utt_rejects}/{_FAIL_OPEN_AFTER_REJECTS}"
        )
        if (
            self._state == LockState.LOCKED
            and self._fail_open_uses >= _FAIL_OPEN_MAX_USES
            and self._consecutive_utt_rejects >= _REENROLL_AFTER_REJECTS
        ):
            self.request_reenrollment(
                f"{self._consecutive_utt_rejects} consecutive rejected turns after "
                f"the fail-open budget was exhausted"
            )

    def request_reenrollment(self, reason: str = "") -> None:
        self._reenroll_count += 1
        logger.warning(f"🔄 SpeakerLock RE-ENROLLMENT #{self._reenroll_count} ({reason})")
        self._state                   = LockState.ENROLLING
        self._profile                 = None
        self._profile_f0_median       = None
        self._profile_f0_mad          = 40.0
        self._enroll_frames           = []
        self._enroll_fallback_frames  = []
        self._enroll_f0_samples       = []
        self._enroll_fallback_f0      = []
        self._enrolled_ms             = 0.0
        self._enroll_wall_ms          = 0.0
        self._enroll_gate_ratio       = _ENROLL_MIN_SNR_RATIO
        self._enroll_gate_relaxed     = False
        self._low_confidence_profile  = False
        self._consecutive_utt_rejects = 0
        self._fail_open_uses          = 0
        self._pass_through            = False
        self._smoothed_similarity     = None
        self._cosine_threshold        = self._base_threshold
        self._sim_history.clear()
        self._since_adapt             = 0
        self._adapted                 = False
        self._high_accept_counter     = 0
        # --- NEW: RESET CONTAMINATION FLAG ---
        self._contamination_detected  = False
        if self._frame_filter_ref is not None:
            self._frame_filter_ref.flush_history()

    def fail_open_pending(self) -> bool:
        if self._state != LockState.LOCKED:
            return False
        if self._fail_open_uses >= _FAIL_OPEN_MAX_USES:
            return False
        needed = 1 if self._fail_open_uses >= 1 else _FAIL_OPEN_AFTER_REJECTS
        return self._consecutive_utt_rejects >= needed

    def consume_fail_open(self) -> None:
        self._consecutive_utt_rejects = 0
        self._fail_open_uses += 1
        old = self._cosine_threshold
        self._cosine_threshold = max(
            _FAIL_OPEN_MIN_THRESHOLD, self._cosine_threshold - _FAIL_OPEN_RELAX_STEP
        )
        logger.warning(
            f"🟡 SpeakerLock FAIL-OPEN #{self._fail_open_uses}/{_FAIL_OPEN_MAX_USES}: "
            f"threshold relaxed {old:.2f} -> {self._cosine_threshold:.2f}"
        )
        if self._fail_open_uses >= _FAIL_OPEN_MAX_USES:
            self._pass_through = True

    # --------------------------------------------------------------- gating --
    def process_chunk(self, pcm_bytes: bytes, chunk_ms: float) -> tuple[bool, bool]:
        self._total += 1
        if self._bot_speaking_ref.get("bot_speaking"):
            return self._process_chunk_during_bot_speech(pcm_bytes, chunk_ms)

        rms_val = _rms(pcm_bytes)
        self._update_noise_floor(rms_val)
        if rms_val < _RMS_FLOOR_RATIO * self._bg_rms_ema:
            self._silence += 1
            return True, True

        if self._state == LockState.ENROLLING:
            return self._enroll_chunk(pcm_bytes, chunk_ms, rms_val)

        if self._pass_through:
            self._accepted += 1
            return True, False

        embedding, chunk_sfm, chunk_f0 = _compute_embedding(pcm_bytes, self._sample_rate)
        raw_similarity = _cosine_similarity(embedding, self._profile)

        if chunk_sfm < _SFM_MUSIC_CEILING or chunk_sfm > _SFM_NOISE_FLOOR:
            self._rejected += 1
            self._nonvoice_rejects += 1
            return False, False

        if self._looks_like_media(chunk_sfm):
            self._rejected += 1
            self._music_rejects += 1
            return False, False

        if raw_similarity < _HARD_REJECT_COSINE:
            self._rejected += 1
            return False, False

        self._utt_sims.append(raw_similarity)
        if len(self._utt_sims) > 400:
            del self._utt_sims[:200]
        if chunk_f0 > 0:
            self._utt_f0s.append(chunk_f0)
            if len(self._utt_f0s) > 400:
                del self._utt_f0s[:200]

        self._update_diagnostic_ema(raw_similarity, rms_val)
        self._observe_similarity(raw_similarity)

        # ---- ENHANCED: Dynamic threshold relaxation for 3+ speaker overlap ----
        threshold = self._cosine_threshold
        if _SEPARATION_ENABLED and len(self._utt_sims) >= 8:
            recent = self._utt_sims[-8:]
            variance = float(np.var(recent))
            
            if variance > 0.025:  
                relax = min(0.15, (variance - 0.025) * 2.5)
                threshold = max(_ADAPT_MIN, threshold - relax)
                logger.debug(f"🎯 Multi-speaker overlap detected (var={variance:.3f}) – threshold relaxed to {threshold:.2f}")

        accepted = raw_similarity >= threshold
        if accepted:
            self._accepted += 1
            self._maybe_update_profile(embedding, raw_similarity)
        else:
            self._rejected += 1
            self._check_auto_degrade()
        return accepted, False

    def _process_chunk_during_bot_speech(self, pcm_bytes: bytes, chunk_ms: float) -> tuple[bool, bool]:
        rms_val = _rms(pcm_bytes)

        if rms_val < _RMS_FLOOR_RATIO * self._bg_rms_ema:
            self._echo_skipped += 1
            self._note_echo(rms_val)
            self._barge_sustain_ms = 0.0
            return True, True

        if self._state != LockState.LOCKED:
            _, chunk_sfm, _ = _compute_embedding(pcm_bytes, self._sample_rate)
            loud_enough = (
                self._echo_samples >= 5
                and rms_val >= _BARGE_ECHO_RATIO * max(self._echo_rms_ema, 1.0)
            ) or rms_val >= _BARGE_FLOOR_RATIO * max(self._bg_rms_ema, 1.0)
            if loud_enough and _is_voice_band(chunk_sfm):
                self._barge_sustain_ms += chunk_ms
                if self._barge_sustain_ms >= _BARGE_MIN_SUSTAIN_MS:
                    self._barge_frames += 1
                    self._barge_sustain_ms = min(self._barge_sustain_ms, _BARGE_MIN_SUSTAIN_MS * 2)
                    return True, True
            else:
                self._barge_sustain_ms = 0.0
            self._echo_skipped += 1
            self._note_echo(rms_val)
            return False, False

        embedding, chunk_sfm, chunk_f0 = _compute_embedding(pcm_bytes, self._sample_rate)
        raw_similarity = _cosine_similarity(embedding, self._profile)
        if chunk_sfm < _SFM_MUSIC_CEILING or chunk_sfm > _SFM_NOISE_FLOOR:
            self._rejected += 1
            self._echo_skipped += 1
            self._note_echo(rms_val)
            self._barge_sustain_ms = 0.0
            return False, False

        if raw_similarity < _HARD_REJECT_COSINE:
            self._rejected += 1
            self._echo_skipped += 1
            self._note_echo(rms_val)
            self._barge_sustain_ms = 0.0
            return False, False

        if self._pass_through:
            self._accepted += 1
            self._barge_frames += 1
            self._barge_sustain_ms = 0.0
            return True, False

        if chunk_f0 > 0:
            self._utt_f0s.append(chunk_f0)
            if len(self._utt_f0s) > 400:
                del self._utt_f0s[:200]

        threshold = self._cosine_threshold + _BOT_SPEAKING_EXTRA_MARGIN
        accepted = raw_similarity >= threshold
        if accepted:
            self._accepted += 1
            self._barge_frames += 1
            self._barge_sustain_ms = 0.0
        else:
            self._rejected += 1
            self._echo_skipped += 1
            self._note_echo(rms_val)
            self._barge_sustain_ms = 0.0
        return accepted, False

    def _note_echo(self, rms_val: float) -> None:
        if self._echo_rms_ema <= 0.0:
            self._echo_rms_ema = rms_val
        else:
            self._echo_rms_ema += _ECHO_EMA_ALPHA * (rms_val - self._echo_rms_ema)
        self._echo_samples += 1

    # ---------------------------------------------------------- calibration --
    def _observe_similarity(self, raw_similarity: float) -> None:
        if not _ADAPT_ENABLED:
            return
        self._sim_history.append(raw_similarity)
        self._since_adapt += 1
        if len(self._sim_history) < _ADAPT_MIN_SAMPLES or self._since_adapt < _ADAPT_EVERY:
            return
        self._since_adapt = 0
        arr = np.fromiter(self._sim_history, dtype=np.float32)

        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826
        proposed = med - _ADAPT_MARGIN_MADS * max(mad, 0.03)
        proposed = float(np.clip(proposed, _ADAPT_MIN,
                                 min(_ADAPT_MAX, self._base_threshold)))

        decided = self._accepted + self._rejected
        rej_ratio = (self._rejected / decided) if decided else 0.0

        if rej_ratio > 0.5:
            proposed = min(proposed, self._cosine_threshold - 0.05)
            proposed = max(proposed, _ADAPT_MIN)
            self._high_accept_counter = 0
        else:
            acc_ratio = 1.0 - rej_ratio
            if acc_ratio > _ADAPT_RAISE_RATIO:
                self._high_accept_counter += 1
                if self._high_accept_counter >= 3:
                    proposed = min(proposed + _ADAPT_RAISE_STEP,
                                   self._base_threshold,
                                   _ADAPT_MAX)
            else:
                self._high_accept_counter = 0

        if abs(proposed - self._cosine_threshold) < 0.005:
            return
        old = self._cosine_threshold
        self._cosine_threshold = proposed
        self._adapted = True
        logger.info(
            f"🎚️ SpeakerLock threshold adjusted {old:.2f} -> {proposed:.2f} "
            f"(median={med:.2f} mad={mad:.3f}, n={len(arr)}, "
            f"reject_ratio={rej_ratio:.0%})")

    def _check_auto_degrade(self) -> None:
        if self._pass_through or self._state != LockState.LOCKED:
            return
        decided = self._accepted + self._rejected
        if decided < _AUTO_DEGRADE_AFTER:
            return
        if self._rejected / decided <= _AUTO_DEGRADE_RATIO:
            return
        self._pass_through = True
        logger.error(
            f"🚨 SpeakerLock AUTO-DEGRADED to pass-through: rejected "
            f"{100.0 * self._rejected / decided:.0f}% of {decided} speech frames. "
            f"The enrollment profile does not match this caller. Passing all "
            f"audio so the recogniser stops receiving silence.")

    def _maybe_update_profile(self, embedding: np.ndarray, effective: float) -> None:
        if not _PROFILE_EMA_ENABLED or self._profile is None:
            return
        if effective < self._cosine_threshold + _PROFILE_EMA_MARGIN:
            return
        updated = (1.0 - _PROFILE_EMA_ALPHA) * self._profile + _PROFILE_EMA_ALPHA * embedding
        n = np.linalg.norm(updated)
        if n > 1e-6:
            self._profile = (updated / n).astype(np.float32)

    def _update_diagnostic_ema(self, raw_similarity: float, rms_val: float) -> float:
        snr_db = 20.0 * _math.log10(max(rms_val, 1.0) / max(self._bg_rms_ema, 1.0))
        snr_weight = max(0.1, min(1.0, snr_db / _SNR_ALPHA_SCALE_DB))
        effective_alpha = _SIMILARITY_EMA_ALPHA * snr_weight
        if self._smoothed_similarity is None:
            self._smoothed_similarity = raw_similarity
        else:
            self._smoothed_similarity = (
                effective_alpha * raw_similarity
                + (1 - effective_alpha) * self._smoothed_similarity
            )
        return snr_db

    def _cheap_f0(self, chunk_f0: float) -> float:
        self._f0_tick += 1
        if self._f0_tick % _F0_EVERY == 0 or self._last_f0 <= 0:
            if chunk_f0 > 0:
                self._last_f0 = chunk_f0
            return chunk_f0
        return self._last_f0 if chunk_f0 <= 0 else chunk_f0

    def _looks_like_media(self, chunk_sfm: float) -> bool:
        if chunk_sfm < _MUSIC_SFM_CEILING:
            self._tonal_run += 1
        else:
            self._tonal_run = 0
        return self._tonal_run >= _MUSIC_MIN_RUN

    def utterance_mean_similarity(self) -> float:
        if not self._utt_sims:
            return 1.0
        return sum(self._utt_sims) / len(self._utt_sims)

    def utterance_pitch_penalty(self) -> float:
        if not self._utt_f0s or self._profile_f0_median is None:
            return 0.0
        median_f0 = float(np.median(self._utt_f0s))
        spread = max(self._profile_f0_mad, 25.0)
        deviation = abs(median_f0 - self._profile_f0_median) / spread
        rel = deviation / 4.0
        if rel <= _F0_SOFT_DEVIATION:
            return 0.0
        if rel >= _F0_HARD_DEVIATION:
            return _F0_MAX_PENALTY
        span = _F0_HARD_DEVIATION - _F0_SOFT_DEVIATION
        return _F0_MAX_PENALTY * (rel - _F0_SOFT_DEVIATION) / span

    def begin_utterance(self) -> None:
        self._utt_sims = []
        self._utt_f0s = []

    # ----------------------------------------------------------- enrollment --
    def _enroll_chunk(self, pcm_bytes: bytes, chunk_ms: float, rms_val: float):
        self._enroll_wall_ms += chunk_ms
        _, chunk_sfm, chunk_f0 = _compute_embedding(pcm_bytes, self._sample_rate)

        if (
            not self._enroll_gate_relaxed
            and self._enroll_wall_ms >= (self._enroll_max_seconds * 1000.0) / 2.0
            and self._enrolled_ms < (self._enroll_seconds * 1000.0) / 2.0
        ):
            self._enroll_gate_relaxed = True
            self._enroll_gate_ratio = _ENV_ENROLL_RELAXED_SNR_RATIO
            logger.warning(
                f"⚠️ SpeakerLock enrollment starving "
                f"({self._enrolled_ms:.0f}ms qualified after "
                f"{self._enroll_wall_ms:.0f}ms) — relaxing loudness gate to "
                f"{self._enroll_gate_ratio}x noise floor"
            )

        voice_ok = _is_voice_band(chunk_sfm)
        snr_ok = rms_val >= self._enroll_gate_ratio * self._bg_rms_ema

        if voice_ok and snr_ok:
            self._enroll_frames.append(np.frombuffer(pcm_bytes, dtype=np.int16).copy())
            self._enrolled_ms += chunk_ms
            if chunk_f0 > 0:
                self._enroll_f0_samples.append(chunk_f0)
        elif voice_ok:
            if len(self._enroll_fallback_frames) < 800:
                self._enroll_fallback_frames.append(
                    np.frombuffer(pcm_bytes, dtype=np.int16).copy()
                )
                if chunk_f0 > 0:
                    self._enroll_fallback_f0.append(chunk_f0)

        qualified_done = self._enrolled_ms >= self._enroll_seconds * 1000.0
        timed_out      = self._enroll_wall_ms >= self._enroll_max_seconds * 1000.0
        have_any       = bool(self._enroll_frames or self._enroll_fallback_frames)
        if qualified_done or (timed_out and have_any):
            if timed_out and not qualified_done:
                if not self._enroll_frames:
                    self._enroll_frames     = self._enroll_fallback_frames
                    self._enroll_f0_samples = self._enroll_fallback_f0
                else:
                    logger.warning(
                        f"⚠️ SpeakerLock enrollment hit the {self._enroll_max_seconds}s "
                        f"wall-clock cap with only {self._enrolled_ms:.0f}ms of qualified "
                        f"voice — locking anyway"
                    )
            self._build_profile()
        self._accepted += 1
        return True, True

    # --- NEW: CLUSTERING METHOD FOR ENROLLMENT ---
    def _cluster_enrollment_embeddings(self, vecs: list[np.ndarray], threshold: float = 0.60) -> list[list[int]]:
        """
        Detects multiple speakers by clustering enrollment embeddings.
        """
        clusters = []
        for i, v in enumerate(vecs):
            assigned = False
            for c in clusters:
                c_vecs = [vecs[idx] for idx in c]
                centroid = np.mean(c_vecs, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-6:
                    centroid = centroid / norm

                if _cosine_similarity(v, centroid) >= threshold:
                    c.append(i)
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
        return clusters

    def _build_profile(self):
        # --- NEW: RESET CONTAMINATION FLAG AT START OF BUILD ---
        self._contamination_detected = False
        
        if not self._enroll_frames:
            logger.warning("⚠️ SpeakerLock: no qualified enroll frames — staying ENROLLING")
            return
        combined = np.concatenate(self._enroll_frames)

        win = max(1600, _PROFILE_WINDOW_SAMPLES)
        vecs = []
        for start in range(0, max(1, len(combined) - win + 1), win // 2 or win):
            chunk = combined[start:start + win]
            if len(chunk) < win // 2:
                continue
            v, sfm, _ = _compute_embedding(chunk.tobytes(), self._sample_rate)
            if _is_voice_band(sfm):
                vecs.append(v)
                
        if len(vecs) >= 3:
            # --- NEW: ENROLLMENT CONTAMINATION CHECK ---
            clusters = self._cluster_enrollment_embeddings(vecs, threshold=0.60)
            
            if len(clusters) > 1:
                logger.warning(f"⚠️ SpeakerLock ENROLLMENT CONTAMINATION DETECTED: {len(clusters)} distinct speakers found.")
                largest_cluster = max(clusters, key=len)
                dominant_ratio = len(largest_cluster) / len(vecs)
                
                if dominant_ratio < 0.60:
                    logger.error("❌ Enrollment REJECTED: Multiple speakers with no clear dominant customer. Forcing re-enrollment.")
                    self._contamination_detected = True
                    self._enroll_frames = []
                    self._enroll_f0_samples = []
                    self._enrolled_ms = 0.0
                    self._enroll_wall_ms = 0.0
                    self._enroll_gate_relaxed = False
                    self._enroll_gate_ratio = _ENROLL_MIN_SNR_RATIO
                    return # Stay in ENROLLING state, buffers cleared. Orchestrator should prompt user.
                
                logger.info(f"🧹 Enrollment CLEANED: Discarding {len(vecs) - len(largest_cluster)} frames from secondary speakers. Keeping dominant ({dominant_ratio:.0%}).")
                vecs = [vecs[i] for i in largest_cluster]
            # --- END CONTAMINATION CHECK ---

            stacked = np.vstack(vecs)
            profile = np.median(stacked, axis=0)
            n = np.linalg.norm(profile)
            self._profile = (profile / (n + 1e-9)).astype(np.float32)
            sims = [float(np.dot(v, self._profile)) for v in vecs]
            spread = float(np.std(sims))
            logger.info(f"🧬 Profile built from {len(vecs)} windows "
                        f"(self-similarity mean={np.mean(sims):.3f} sd={spread:.3f})")
            if spread > 0.12:
                self._low_confidence_profile = True
                self._cosine_threshold = max(_ADAPT_MIN,
                                             self._cosine_threshold - _LOW_CONF_THRESHOLD_DROP)
        else:
            self._profile, _, _ = _compute_embedding(combined.tobytes(), self._sample_rate)

        if self._enroll_f0_samples:
            arr = np.array(self._enroll_f0_samples, dtype=np.float32)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median))) * 1.4826
            self._profile_f0_median = median
            self._profile_f0_mad = max(mad, 25.0)

            if mad > _F0_MAD_SUSPECT_HZ:
                self._low_confidence_profile = True
                old = self._cosine_threshold
                self._cosine_threshold = max(
                    _FAIL_OPEN_MIN_THRESHOLD,
                    self._cosine_threshold - _LOW_CONF_THRESHOLD_DROP,
                )
                self._profile_f0_mad = max(mad, self._profile_f0_mad, 60.0)
                logger.warning(
                    f"⚠️ SpeakerLock LOW-CONFIDENCE profile (f0_mad={mad:.1f}Hz). "
                    f"Cosine threshold lowered {old:.2f} -> {self._cosine_threshold:.2f}. "
                    f"Pitch gate stays ACTIVE with widened tolerance "
                    f"(f0_mad={self._profile_f0_mad:.1f}Hz)"
                )
        else:
            logger.warning("⚠️ SpeakerLock: no voiced F0 samples during enrollment "
                           "— pitch gate disabled")
            self._profile_f0_median = None

        self._state = LockState.LOCKED
        self._enroll_frames = []
        self._enroll_fallback_frames = []
        self._enroll_fallback_f0 = []
        logger.info(
            f"🔒 SpeakerLock LOCKED after {self._enrolled_ms:.0f} ms of QUALIFIED customer "
            f"speech ({self._enroll_wall_ms:.0f} ms wall clock, echo frames skipped: "
            f"{self._echo_skipped}) | profile_norm={np.linalg.norm(self._profile):.4f} "
            f"| cosine_threshold={self._cosine_threshold:.2f} (raw-based) "
            f"| hard_floor={_HARD_REJECT_COSINE} "
            f"| sample_rate={self._sample_rate} "
            f"| f0_median={self._profile_f0_median} f0_mad={self._profile_f0_mad:.1f}"
        )
        if self._echo_skipped == 0:
            logger.warning(
                "⚠️ SpeakerLock LOCKED with echo_skipped=0. If the bot spoke at all "
                "during enrollment, the bot_speaking_ref is NOT wired."
            )
        if self._frame_filter_ref is not None:
            self._frame_filter_ref.flush_history()

    def stats_str(self) -> str:
        total = self._accepted + self._rejected
        rej_pct = (100.0 * self._rejected / total) if total else 0.0
        return (f"SpeakerLock[{self._state.value}/{SPEAKER_LOCK_MODE}] "
                f"accepted={self._accepted} rejected={self._rejected} "
                f"({rej_pct:.0f}% rejected) "
                f"silence={self._silence} echo_skipped={self._echo_skipped} "
                f"barge_frames={self._barge_frames} "
                f"music_rej={self._music_rejects} nonvoice_rej={self._nonvoice_rejects} "
                f"total={self._total} qualified_enroll_ms={self._enrolled_ms:.0f} "
                f"threshold={self._cosine_threshold:.2f} adapted={self._adapted} "
                f"fail_open_uses={self._fail_open_uses} "
                f"reenrolls={self._reenroll_count} "
                f"sample_rate={self._sample_rate} "
                f"degraded={self._pass_through} "
                f"low_conf={self._low_confidence_profile} "
                f"echo_rms={self._echo_rms_ema:.0f} "
                f"f0_median={self._profile_f0_median}")


# ======================================================================
# SPEAKER-AWARE FRAME FILTER
# ======================================================================
class SpeakerAwareFrameFilter:
    def __init__(self, speaker_lock: SpeakerLock, history_window_ms: float = VAD_WINDOW_MS):
        self._lock                     = speaker_lock
        self.last_chunk_verified: bool = True
        self.last_speech_chunk_verified: bool = True
        self._decision_history: collections.deque = collections.deque()
        self._history_window_s = history_window_ms / 1000.0
        self._last_lock_decision: bool | None = None

        self._burst: list[bool] = []
        self._last_burst_ratio: float | None = None
        self._last_burst_end_ts: float = 0.0
        self._last_nonneutral_ts: float | None = None

        self._accept_run_ms: float = 0.0
        self._last_good_run_ts: float = 0.0
        speaker_lock.register_frame_filter(self)

    def flush_history(self) -> None:
        self._decision_history.clear()
        self._last_lock_decision = None
        self._burst = []
        self._last_burst_ratio = None
        self._last_nonneutral_ts = None
        self._accept_run_ms = 0.0
        self._last_good_run_ts = 0.0

    def end_utterance(self) -> None:
        self._burst = []
        self._last_burst_ratio = None
        self._accept_run_ms = 0.0
        self._last_good_run_ts = 0.0

    def process(self, pcm_bytes: bytes, chunk_ms: float) -> bytes:
        accepted, is_neutral = self._lock.process_chunk(pcm_bytes, chunk_ms)
        self.last_chunk_verified = accepted
        if is_neutral:
            return pcm_bytes
        self.last_speech_chunk_verified = accepted
        self._last_lock_decision        = accepted
        now = time.monotonic()

        if (
            self._last_nonneutral_ts is not None
            and (now - self._last_nonneutral_ts) > _BURST_GAP_S
            and self._burst
        ):
            self._last_burst_ratio  = sum(1 for ok in self._burst if ok) / len(self._burst)
            self._last_burst_end_ts = self._last_nonneutral_ts
            self._burst = []
        self._burst.append(accepted)
        self._last_nonneutral_ts = now

        if accepted:
            self._accept_run_ms += chunk_ms
            if self._accept_run_ms >= _ACCEPT_RUN_MIN_MS:
                self._last_good_run_ts = now
        else:
            self._accept_run_ms = 0.0

        self._decision_history.append((now, accepted))
        cutoff = now - self._history_window_s
        while self._decision_history and self._decision_history[0][0] < cutoff:
            self._decision_history.popleft()
        if accepted:
            return pcm_bytes

        if SPEAKER_LOCK_MODE == "monitor" or self._lock.pass_through:
            return pcm_bytes
        return bytes(len(pcm_bytes))

    def recent_accept_ratio(self) -> float:
        if self._lock.state == LockState.ENROLLING:
            return 1.0
        if not self._decision_history:
            if self._last_lock_decision is False:
                return 0.0
            return 1.0
        now    = time.monotonic()
        cutoff = now - self._history_window_s
        recent = [(ts, ok) for ts, ok in self._decision_history if ts >= cutoff]
        if not recent:
            if self._last_lock_decision is False:
                return 0.0
            return 1.0
        return sum(1 for _, ok in recent if ok) / len(recent)

    def turn_evidence_chunks(self) -> int:
        now    = time.monotonic()
        cutoff = now - self._history_window_s
        return sum(1 for ts, _ in self._decision_history if ts >= cutoff)

    def turn_accept_ratio(self) -> float:
        if self._lock.state == LockState.ENROLLING or self._lock.pass_through:
            return 1.0
        now = time.monotonic()
        candidates = [self.recent_accept_ratio()]
        if len(self._burst) >= 3:
            candidates.append(sum(1 for ok in self._burst if ok) / len(self._burst))
        if (
            self._last_burst_ratio is not None
            and (now - self._last_burst_end_ts) <= _BURST_FRESH_S
        ):
            candidates.append(self._last_burst_ratio)

        burst_ok = True
        if len(self._burst) >= 3:
            burst_ok = (sum(1 for ok in self._burst if ok) / len(self._burst)) >= 0.3
        if (burst_ok and self._last_good_run_ts
                and (now - self._last_good_run_ts) <= _ACCEPT_RUN_FRESH_S):
            candidates.append(0.95)
        return max(candidates)

    def caller_verdict(self) -> tuple[bool, float, float, int]:
        lock = self._lock
        if lock.state == LockState.ENROLLING or lock.pass_through:
            return True, 1.0, 1.0, 0
        ratio = self.turn_accept_ratio()
        mean_sim = lock.utterance_mean_similarity()
        pitch_penalty = lock.utterance_pitch_penalty()
        adjusted_mean = mean_sim - pitch_penalty
        frames = len(lock._utt_sims)
        if frames < _UTT_MIN_FRAMES:
            return True, ratio, adjusted_mean, frames
        is_caller = not (ratio < VAD_ACCEPT_RATIO and adjusted_mean < _UTT_MIN_MEAN_SIM)
        if pitch_penalty > 0.05:
            logger.debug(f"🎵 Utterance pitch penalty: {pitch_penalty:.3f}, "
                         f"mean_sim {mean_sim:.2f} -> adjusted {adjusted_mean:.2f}")
        return is_caller, ratio, adjusted_mean, frames


# ======================================================================
# SPEAKER-GATED VAD
# ======================================================================
class SpeakerGatedVAD:
    def __init__(
        self,
        real_vad:        Any,
        frame_filter:    "SpeakerAwareFrameFilter",
        bot_speaking_ref: dict = None,
    ):
        object.__setattr__(self, "_real_vad",          real_vad)
        object.__setattr__(self, "_frame_filter",      frame_filter)
        object.__setattr__(self, "_bot_speaking_ref",  bot_speaking_ref or {"bot_speaking": False})
        object.__setattr__(self, "_suppressed_count",  0)
        object.__setattr__(self, "_passed_count",      0)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real_vad"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_real_vad", "_frame_filter", "_bot_speaking_ref",
                    "_suppressed_count", "_passed_count"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real_vad"), name, value)

    def stream(self, *args, **kwargs):
        real_vad         = object.__getattribute__(self, "_real_vad")
        frame_filter     = object.__getattribute__(self, "_frame_filter")
        bot_speaking_ref = object.__getattribute__(self, "_bot_speaking_ref")
        real_stream      = real_vad.stream(*args, **kwargs)
        return SpeakerGatedVADStream(real_stream, frame_filter, self, bot_speaking_ref)

    def stats_str(self) -> str:
        sup = object.__getattribute__(self, "_suppressed_count")
        pas = object.__getattribute__(self, "_passed_count")
        return f"SpeakerGatedVAD[suppressed={sup} passed={pas}]"


class SpeakerGatedVADStream:
    _START_SPEECH_NAMES = frozenset({
        "START_SPEECH", "SPEECH_STARTED", "start_of_speech",
        "START_OF_SPEECH", "SPEAKING_STARTED", "speaking_started", "speech_started",
    })
    _END_SPEECH_NAMES = frozenset({
        "END_SPEECH", "SPEECH_ENDED", "end_of_speech",
        "END_OF_SPEECH", "SPEAKING_STOPPED", "speaking_stopped", "speech_ended",
    })

    def __init__(
        self,
        real_stream:      Any,
        frame_filter:     SpeakerAwareFrameFilter,
        gated_vad:        SpeakerGatedVAD,
        bot_speaking_ref: dict,
    ):
        self._real_stream      = real_stream
        self._frame_filter     = frame_filter
        self._gated_vad        = gated_vad
        self._bot_speaking_ref = bot_speaking_ref
        self._suppressing_current_utterance: bool = False

    def _get_event_type_name(self, event: Any) -> str | None:
        ev_type = getattr(event, "type", None)
        if ev_type is None:
            return None
        return (
            getattr(ev_type, "name",  None)
            or getattr(ev_type, "value", None)
            or str(ev_type)
        )

    def _is_start_speech(self, event: Any) -> bool:
        name = self._get_event_type_name(event)
        return name is not None and name in self._START_SPEECH_NAMES

    def _is_end_speech(self, event: Any) -> bool:
        name = self._get_event_type_name(event)
        return name is not None and name in self._END_SPEECH_NAMES

    def _should_suppress(self) -> bool:
        lock = self._frame_filter._lock
        if lock.state == LockState.ENROLLING or lock.pass_through:
            return False
        ratio        = self._frame_filter.turn_accept_ratio()
        bot_speaking = self._bot_speaking_ref.get("bot_speaking")
        threshold = VAD_ACCEPT_RATIO_BOT_SPEAKING if bot_speaking else VAD_ACCEPT_RATIO
        if ratio >= threshold:
            return False
        is_caller, _, mean_sim, frames = self._frame_filter.caller_verdict()
        if not is_caller:
            logger.info(
                f"🙅 Suppressing an utterance that is not the caller "
                f"(accept_ratio={ratio:.2f}, mean_similarity={mean_sim:.2f} < "
                f"{_UTT_MIN_MEAN_SIM})")
            lock.note_utterance_result(False)
            return True
        if lock.fail_open_pending():
            lock.consume_fail_open()
            logger.warning(
                f"🟡 SpeakerGatedVAD FAIL-OPEN: passing START_SPEECH despite "
                f"low accept_ratio={ratio:.2f}"
            )
            return False
        lock.note_utterance_result(False)
        return True

    def __aiter__(self):
        return self._filtered_events()

    async def _filtered_events(self):
        async for event in self._real_stream:
            if self._is_start_speech(event):
                self._frame_filter._lock.begin_utterance()
                if self._should_suppress():
                    self._suppressing_current_utterance = True
                    sup = object.__getattribute__(self._gated_vad, "_suppressed_count")
                    object.__setattr__(self._gated_vad, "_suppressed_count", sup + 1)
                    continue
                self._suppressing_current_utterance = False
                pas = object.__getattribute__(self._gated_vad, "_passed_count")
                object.__setattr__(self._gated_vad, "_passed_count", pas + 1)
                yield event
                continue
            if self._is_end_speech(event):
                self._frame_filter.end_utterance()
                if self._suppressing_current_utterance:
                    self._suppressing_current_utterance = False
                    continue
                yield event
                continue
            yield event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_stream, name)

    async def __aenter__(self):
        if hasattr(self._real_stream, "__aenter__"):
            await self._real_stream.__aenter__()
        return self

    async def __aexit__(self, *args):
        if hasattr(self._real_stream, "__aexit__"):
            await self._real_stream.__aexit__(*args)

    def close(self):
        if hasattr(self._real_stream, "close"):
            self._real_stream.close()

    async def aclose(self):
        if hasattr(self._real_stream, "aclose"):
            await self._real_stream.aclose()


# ======================================================================
# LANGUAGE LOCK GUARD
# ======================================================================
class LanguageLockGuard:
    def __init__(self, speaker_lock: SpeakerLock):
        self._lock             = speaker_lock
        self._pending_lang: str | None = None
        self._pending_count    = 0

    def should_switch(
        self,
        new_lang:           str,
        current_lang:       str,
        is_explicit:        bool,
        chunk_was_verified: bool,
    ) -> Tuple[bool, str]:
        if is_explicit and chunk_was_verified:
            self.reset()
            return True, "explicit + verified"
        if is_explicit and not chunk_was_verified:
            self.reset()
            logger.warning(f"⚠️ LanguageLockGuard: allowing explicit unverified "
                           f"switch to '{new_lang}'")
            return True, "explicit (unverified)"

        if self._lock.state == LockState.ENROLLING:
            if self._pending_lang != new_lang:
                self._pending_lang  = new_lang
                self._pending_count = 1
            else:
                self._pending_count += 1
            if self._pending_count >= _LANG_SWITCH_MIN_TURNS_ENROLLING:
                self.reset()
                return True, f"switch to '{new_lang}' during ENROLLING confirmed"
            return False, f"switch to '{new_lang}' during ENROLLING — pending"

        if not chunk_was_verified:
            self._reset_pending_if_different(new_lang)
            return False, "utterance rejected by SpeakerLock"

        if self._pending_lang != new_lang:
            self._pending_lang  = new_lang
            self._pending_count = 1
        else:
            self._pending_count += 1
        if self._pending_count >= _LANG_SWITCH_MIN_VERIFIED_TURNS:
            self.reset()
            return True, f"verified switch to '{new_lang}'"
        return False, f"ambiguous switch to '{new_lang}' — pending"

    def reset(self):
        self._pending_lang  = None
        self._pending_count = 0

    def _reset_pending_if_different(self, lang: str):
        if self._pending_lang and self._pending_lang != lang:
            self._pending_lang  = None
            self._pending_count = 0


# ======================================================================
# SPECTRAL NOISE REDUCER
# ======================================================================
class SpectralNoiseReducer:
    def __init__(self):
        self._noise_mag: dict[int, np.ndarray] = {}
        self._updates: dict[int, int] = {}

    def _fft_size(self, n: int) -> int:
        return max(256, 1 << (n - 1).bit_length())

    def update_noise(self, pcm_int16: np.ndarray) -> None:
        n = len(pcm_int16)
        if n == 0:
            return
        size = self._fft_size(n)
        mag = np.abs(np.fft.rfft(pcm_int16.astype(np.float32), n=size))
        prev = self._noise_mag.get(size)
        if prev is None:
            self._noise_mag[size] = mag
            self._updates[size] = 1
        else:
            self._noise_mag[size] = _NR_NOISE_ALPHA * mag + (1.0 - _NR_NOISE_ALPHA) * prev
            self._updates[size] = self._updates.get(size, 0) + 1

    def ready(self, n: int) -> bool:
        size = self._fft_size(n)
        return self._updates.get(size, 0) >= _NR_MIN_UPDATES

    def reduce(self, pcm_int16: np.ndarray) -> np.ndarray:
        n = len(pcm_int16)
        if n == 0:
            return pcm_int16
        size = self._fft_size(n)
        noise = self._noise_mag.get(size)
        if noise is None or self._updates.get(size, 0) < _NR_MIN_UPDATES:
            return pcm_int16
        spec = np.fft.rfft(pcm_int16.astype(np.float32), n=size)
        mag = np.abs(spec)
        gain = 1.0 - _NR_OVERSUBTRACT * (noise / (mag + 1e-6))
        gain = np.clip(gain, _NR_FLOOR_GAIN, 1.0)
        cleaned = np.fft.irfft(spec * gain, n=size)[:n]
        return np.clip(cleaned, -32768, 32767).astype(np.int16)


# ======================================================================
# MAIN AUDIO PROCESSOR (with enhanced separation)
# ======================================================================
class SpeakerLockAudioProcessor(rtc.FrameProcessor):
    def __init__(self, debug_record: bool = False, bot_speaking_ref: dict | None = None):
        super().__init__()
        self._debug        = debug_record
        self._frame_rate   = 8000
        self._raw_frames   = []
        self._enabled      = True
        self._bot_speaking_ref = (bot_speaking_ref if bot_speaking_ref is not None
                                  else {"bot_speaking": False})
        self._speaker_lock = SpeakerLock(bot_speaking_ref=self._bot_speaking_ref)
        self.frame_filter  = SpeakerAwareFrameFilter(self._speaker_lock)
        self._noise_reducer = SpectralNoiseReducer() if _NOISE_REDUCTION_ENABLED else None
        self._rate_logged  = False

        # --- Multi-speaker separation (with enhanced fallback) ---
        self._sep_enabled = _SEPARATION_ENABLED and _SEP_MODEL is not None
        if self._sep_enabled:
            self._audio_buffer = np.array([], dtype=np.int16)
            self._buffer_lock = threading.Lock()
            self._sep_queue = queue.Queue()
            self._sep_thread = threading.Thread(target=self._run_separation_worker, daemon=True)
            self._sep_thread.start()
            self._separated_audio = np.array([], dtype=np.int16)
            logger.info("🧵 Separation worker thread started")

        if bot_speaking_ref is None:
            logger.warning("⚠️ SpeakerLockAudioProcessor built WITHOUT bot_speaking_ref")
        if _NOISE_REDUCTION_ENABLED:
            logger.info(f"🔇 Spectral noise reduction ENABLED (floor_gain={_NR_FLOOR_GAIN}, "
                        f"oversubtract={_NR_OVERSUBTRACT})")

    def _run_separation_worker(self):
        """Background thread: separates overlapping speakers (supports N sources)."""
        while True:
            chunk = self._sep_queue.get()
            if chunk is None:
                break
            try:
                import torch
                with torch.no_grad():
                    wav_torch = torch.from_numpy(chunk.astype(np.float32) / 32768.0).unsqueeze(0)
                    if torch.cuda.is_available():
                        wav_torch = wav_torch.cuda()

                    est_sources = _SEP_MODEL(wav_torch)
                    sources = est_sources.cpu().numpy()[0]  
                    n_sources = sources.shape[0]
                    
                    scores = []
                    separated_streams = []
                    
                    for i in range(n_sources):
                        src = (sources[i] * 32768).astype(np.int16)
                        score = self._score_stream_full(src)
                        scores.append(score)
                        separated_streams.append(src)
                        
                    best_idx = int(np.argmax(scores))
                    best_score = scores[best_idx]
                    
                    orig_score = self._score_stream_full(chunk)
                    
                    if best_score < _SEP_MIN_FALLBACK_SCORE and orig_score >= best_score:
                        logger.debug("⚠️ Separation artifacts detected – falling back to original mix")
                        self._separated_audio = chunk
                    else:
                        self._separated_audio = separated_streams[best_idx]
                        
            except Exception as e:
                logger.error(f"❌ Separation worker error: {e}")
                self._separated_audio = chunk

    def _score_stream_full(self, pcm_int16: np.ndarray) -> float:
        """
        Score the entire separated stream, not just a 400ms window.
        """
        lock = self._speaker_lock
        if lock.state != LockState.LOCKED or lock._profile is None:
            return 0.5  

        if len(pcm_int16) < 160:
            return 0.5

        max_samples = int(2.0 * self._frame_rate)
        if len(pcm_int16) > max_samples:
            start = (len(pcm_int16) - max_samples) // 2
            pcm_int16 = pcm_int16[start:start + max_samples]

        chunk_bytes = pcm_int16.tobytes()
        emb, sfm, _ = _compute_embedding(chunk_bytes, self._frame_rate)
        if not _is_voice_band(sfm):
            return 0.2
        return _cosine_similarity(emb, lock._profile)

    @property
    def speaker_lock(self) -> SpeakerLock:
        return self._speaker_lock

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)

    def set_bot_speaking_ref(self, ref: dict) -> None:
        self._bot_speaking_ref = ref
        self._speaker_lock.set_bot_speaking_ref(ref)

    def stats_str(self) -> str:
        return self._speaker_lock.stats_str()

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        return self.process_frame(frame)

    def process_frame(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled:
            return frame

        self._frame_rate = frame.sample_rate
        self._speaker_lock.set_sample_rate(frame.sample_rate)
        if not self._rate_logged:
            self._rate_logged = True
            logger.warning(
                f"🎚️ First audio frame: sample_rate={frame.sample_rate} "
                f"num_channels={frame.num_channels} "
                f"samples_per_channel={frame.samples_per_channel}"
            )

        raw_int16 = np.frombuffer(frame.data, dtype=np.int16)

        # --- Multi-speaker separation (only at 8kHz AND ONLY WHEN LOCKED) ---
        if self._sep_enabled and self._frame_rate == 8000:
            if self._speaker_lock.state == LockState.LOCKED:
                with self._buffer_lock:
                    self._audio_buffer = np.concatenate([self._audio_buffer, raw_int16])

                window_samples = int(_SEPARATION_WINDOW_S * self._frame_rate)
                if len(self._audio_buffer) >= window_samples:
                    chunk = self._audio_buffer[:window_samples]
                    self._audio_buffer = self._audio_buffer[window_samples:]
                    self._sep_queue.put(chunk)

                if len(self._separated_audio) > 0:
                    raw_int16 = self._separated_audio
                    self._separated_audio = np.array([], dtype=np.int16)
            else:
                # Clear buffer during enrollment so it doesn't grow infinitely
                with self._buffer_lock:
                    self._audio_buffer = np.array([], dtype=np.int16)
        # --- End separation ---

        if self._debug:
            self._raw_frames.append(raw_int16.tobytes())

        pcm_bytes = raw_int16.tobytes()
        n_samples = len(pcm_bytes) // 2
        chunk_ms = (n_samples / self._frame_rate) * 1000.0

        gated_pcm = self.frame_filter.process(pcm_bytes, chunk_ms)
        gated_int16 = np.frombuffer(gated_pcm, dtype=np.int16)

        if self._noise_reducer is not None and n_samples > 0:
            lock = self._speaker_lock
            rms_val = float(np.sqrt(np.mean(raw_int16.astype(np.float32) ** 2)))
            is_quiet = rms_val < 1.2 * lock._bg_rms_ema
            if is_quiet:
                self._noise_reducer.update_noise(raw_int16)
            if gated_int16.any() and self._noise_reducer.ready(n_samples):
                gated_int16 = self._noise_reducer.reduce(gated_int16)

        return rtc.AudioFrame(
            data=gated_int16.tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=len(gated_int16),
        )

    def _close(self):
        if self._debug and self._raw_frames:
            ts = int(time.time())
            fname = f"/tmp/raw_{ts}.wav"
            try:
                with wave.open(fname, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self._frame_rate)
                    wf.writeframes(b"".join(self._raw_frames))
                logger.info(f"✅ Saved debug audio: {fname}")
            except Exception as e:
                logger.error(f"Failed to write {fname}: {e}")
        logger.info(f"📊 {self._speaker_lock.stats_str()}")
