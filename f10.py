"""
speaker_lock.py - ULTRA-LOW LATENCY & STRICT OVERLAP REJECTION
=============================================================
Optimizations:
- Sample-and-Hold Architecture (Skips heavy AI on intermediate frames)
- Reduced buffer (100ms) for ultra-fast response
- STRICT Overlap Rejection (Prevents loud intruders from hijacking)
- 400ms Separation Window (Eliminates 2-second latency gap)
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
# TUNING - OPTIMIZED FOR REAL-TIME
# ======================================================================
_ENV_SFM_NOISE_FLOOR   = float(os.getenv("SFM_NOISE_FLOOR", "0.88"))
_ENV_SFM_MUSIC_CEILING = float(os.getenv("SFM_MUSIC_CEILING", "0.06"))

_ENV_COSINE_THRESHOLD  = float(os.getenv("COSINE_THRESHOLD", "0.50"))
_ENV_TURN_VERIFY_RATIO = float(os.getenv("TURN_VERIFY_MIN_RATIO", "0.45"))

_HARD_REJECT_COSINE = float(os.getenv("HARD_REJECT_COSINE", "0.28"))

_ENV_F0_SOFT_DEVIATION = float(os.getenv("F0_SOFT_DEVIATION", "0.30"))
_ENV_F0_HARD_DEVIATION = float(os.getenv("F0_HARD_DEVIATION", "0.60"))
_ENV_F0_MAX_PENALTY    = float(os.getenv("F0_MAX_PENALTY", "0.12"))

_ENV_ENROLL_MIN_SNR_RATIO     = float(os.getenv("ENROLL_MIN_SNR_RATIO", "2.0"))
_ENV_ENROLL_RELAXED_SNR_RATIO = float(os.getenv("ENROLL_RELAXED_SNR_RATIO", "1.6"))

_FAIL_OPEN_AFTER_REJECTS  = int(os.getenv("FAIL_OPEN_AFTER_REJECTS", "3"))
_FAIL_OPEN_RELAX_STEP     = float(os.getenv("FAIL_OPEN_RELAX_STEP", "0.05"))
_FAIL_OPEN_MIN_THRESHOLD  = float(os.getenv("FAIL_OPEN_MIN_THRESHOLD", "0.45"))
_FAIL_OPEN_MAX_USES       = int(os.getenv("FAIL_OPEN_MAX_USES", "3"))
_REENROLL_AFTER_REJECTS   = int(os.getenv("REENROLL_AFTER_REJECTS", "4"))

_F0_MAD_SUSPECT_HZ        = float(os.getenv("F0_MAD_SUSPECT_HZ", "60"))
_LOW_CONF_THRESHOLD_DROP  = float(os.getenv("LOW_CONF_THRESHOLD_DROP", "0.05"))
_F0_EVERY = max(1, int(os.getenv("SPEAKER_F0_EVERY", "3")))

_BARGE_ECHO_RATIO   = float(os.getenv("BARGE_ECHO_RATIO", "2.2"))
_BARGE_FLOOR_RATIO  = float(os.getenv("BARGE_FLOOR_RATIO", "3.0"))
_ECHO_EMA_ALPHA     = float(os.getenv("ECHO_EMA_ALPHA", "0.10"))
_BOT_SPEAKING_EXTRA_MARGIN = float(os.getenv("BOT_SPEAKING_EXTRA_MARGIN", "0.04"))
_BARGE_MIN_SUSTAIN_MS = float(os.getenv("BARGE_MIN_SUSTAIN_MS", "150"))

SPEAKER_LOCK_MODE = os.getenv("SPEAKER_LOCK_MODE", "monitor").strip().lower()
if SPEAKER_LOCK_MODE not in ("monitor", "gate"):
    SPEAKER_LOCK_MODE = "monitor"

_AUTO_DEGRADE_AFTER   = int(os.getenv("SPEAKER_AUTO_DEGRADE_AFTER", "150"))
_AUTO_DEGRADE_RATIO   = float(os.getenv("SPEAKER_AUTO_DEGRADE_RATIO", "0.55"))

_ADAPT_ENABLED      = os.getenv("SPEAKER_ADAPT_ENABLED", "true").lower() == "true"
_ADAPT_MIN_SAMPLES  = int(os.getenv("SPEAKER_ADAPT_MIN_SAMPLES", "40"))
_ADAPT_EVERY        = int(os.getenv("SPEAKER_ADAPT_EVERY", "25"))
_ADAPT_MARGIN_MADS  = float(os.getenv("SPEAKER_ADAPT_MARGIN_MADS", "2.5"))
_ADAPT_MIN          = float(os.getenv("SPEAKER_ADAPT_MIN", "0.35"))
_ADAPT_MAX          = float(os.getenv("SPEAKER_ADAPT_MAX", "0.72"))
_ADAPT_RAISE_RATIO   = float(os.getenv("SPEAKER_ADAPT_RAISE_RATIO", "0.92"))
_ADAPT_RAISE_STEP    = float(os.getenv("SPEAKER_ADAPT_RAISE_STEP", "0.005"))

_PROFILE_EMA_ENABLED = os.getenv("SPEAKER_PROFILE_EMA", "true").lower() == "true"
_PROFILE_EMA_ALPHA   = float(os.getenv("SPEAKER_PROFILE_EMA_ALPHA", "0.02"))
_PROFILE_EMA_MARGIN  = float(os.getenv("SPEAKER_PROFILE_EMA_MARGIN", "0.06"))

_UTT_MIN_MEAN_SIM   = float(os.getenv("UTT_MIN_MEAN_SIM", "0.42"))
_UTT_MIN_FRAMES     = int(os.getenv("UTT_MIN_FRAMES", "6"))
_MUSIC_SFM_CEILING  = float(os.getenv("MUSIC_SFM_CEILING", "0.04"))
_MUSIC_MIN_RUN      = int(os.getenv("MUSIC_MIN_RUN", "25"))

_NOISE_REDUCTION_ENABLED = os.getenv("NOISE_REDUCTION", "false").lower() == "true"
_NR_FLOOR_GAIN   = float(os.getenv("NR_FLOOR_GAIN", "0.22"))
_NR_OVERSUBTRACT = float(os.getenv("NR_OVERSUBTRACT", "1.4"))
_NR_NOISE_ALPHA  = float(os.getenv("NR_NOISE_ALPHA", "0.06"))
_NR_MIN_UPDATES  = int(os.getenv("NR_MIN_UPDATES", "8"))

_SEPARATION_ENABLED = os.getenv("SPEAKER_SEPARATION", "false").lower() == "true"
# 🔥 FIX: Reduced from 2.0 to 0.4 seconds to eliminate the 2-second latency gap
_SEPARATION_WINDOW_S = float(os.getenv("SEPARATION_WINDOW_S", "0.4")) 
_SEPARATION_MODEL = os.getenv("SEPARATION_MODEL", "speechbrain/sepformer-libri3mix")
_SEP_MIN_FALLBACK_SCORE = float(os.getenv("SEP_MIN_FALLBACK_SCORE", "0.45"))

ENROLL_SECONDS      = float(os.getenv("ENROLL_SECONDS", "3.0"))
ENROLL_MAX_SECONDS  = float(os.getenv("ENROLL_MAX_SECONDS", "12.0"))
_SAMPLE_RATE         = 8000
_FRAME_SIZE          = 160
_PROFILE_WINDOW_SAMPLES = int(os.getenv("PROFILE_WINDOW_SAMPLES", "2400"))

# ===== ULTRA-LOW LATENCY OPTIMIZATIONS =====
_EMBEDDING_BUFFER_MS = int(os.getenv("EMBEDDING_BUFFER_MS", "100"))
_EMBEDDING_BUFFER_CHUNKS = max(1, _EMBEDDING_BUFFER_MS // 20)
_SIMILARITY_SMOOTH_ALPHA = float(os.getenv("SIMILARITY_SMOOTH_ALPHA", "0.35"))
_SNR_ADAPTIVE_ENABLED = os.getenv("SNR_ADAPTIVE", "true").lower() == "true"
_SNR_GOOD_DB = 15.0
_SNR_BAD_DB = 5.0
_SNR_THRESHOLD_RELAX = 0.08
_MIN_ENROLL_QUALITY = float(os.getenv("MIN_ENROLL_QUALITY", "0.3"))
_FAST_PATH_THRESHOLD = float(os.getenv("FAST_PATH_THRESHOLD", "0.65"))

_RMS_FLOOR_RATIO      = float(os.getenv("RMS_FLOOR_RATIO", "0.9"))
_COSINE_THRESHOLD     = _ENV_COSINE_THRESHOLD
_HARD_REJECT_COSINE   = float(os.getenv("HARD_REJECT_COSINE", "0.28"))
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
VAD_ACCEPT_RATIO_BOT_SPEAKING = float(os.getenv("VAD_ACCEPT_RATIO_BOT_SPEAKING", "0.65"))

TURN_SPEAKER_VERIFY_MIN_RATIO = _ENV_TURN_VERIFY_RATIO

_BURST_GAP_S   = 0.6
_BURST_FRESH_S = float(os.getenv("BURST_FRESH_S", "1.2"))

_ACCEPT_RUN_MIN_MS  = float(os.getenv("ACCEPT_RUN_MIN_MS", "400"))
_ACCEPT_RUN_FRESH_S = float(os.getenv("ACCEPT_RUN_FRESH_S", "1.5"))

N_FILTERS = 20
N_CEPS    = 13

# Separation model loading
_SEP_MODEL = None
if _SEPARATION_ENABLED:
    try:
        import torch
        from asteroid.models import BaseModel
        logger.warning(f"🔊 Separation ENABLED: {_SEPARATION_MODEL} (Window: {_SEPARATION_WINDOW_S}s)")
        _SEP_MODEL = BaseModel.from_pretrained(_SEPARATION_MODEL)
        _SEP_MODEL.eval()
        if torch.cuda.is_available():
            _SEP_MODEL.cuda()
        _SEPARATION_ENABLED = True
    except Exception as e:
        logger.error(f"❌ Separation failed to load: {e}")
        _SEPARATION_ENABLED = False
else:
    logger.info("🔇 Separation DISABLED (relying on ECAPA-TDNN for strict rejection)")

logger.warning(f"🔊 SpeakerLock MODE={SPEAKER_LOCK_MODE.upper()}")
logger.info(
    f"🔊 Config: COSINE={_ENV_COSINE_THRESHOLD}, "
    f"HARD_REJECT={_HARD_REJECT_COSINE}, "
    f"BUFFER={_EMBEDDING_BUFFER_MS}ms (ULTRA FAST)"
)


class LockState(enum.Enum):
    ENROLLING = "ENROLLING"
    LOCKED    = "LOCKED"


# ======================================================================
# AUDIO FEATURE EXTRACTION
# ======================================================================
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
        return np.zeros(N_CEPS - 1 + 4, dtype=np.float32), 0.5, 0.0, 0.0, 0.0
    lfe   = _log_filterbank_energy(pcm, n_filters=N_FILTERS)
    ceps  = _cepstral_coeffs(lfe, n_ceps=N_CEPS)
    zcr      = _zero_crossing_rate(pcm)
    centroid = _spectral_centroid(pcm, sample_rate) / (sample_rate / 2)
    f0_hz    = _estimate_f0(pcm, sample_rate=sample_rate)
    f0_norm  = np.array([min(max(f0_hz, 0.0), _F0_MAX_HZ) / _F0_MAX_HZ], dtype=np.float32)
    sfm_val  = _spectral_flatness(pcm)
    sfm      = np.array([sfm_val], dtype=np.float32)
    zcr_arr  = np.array([zcr], dtype=np.float32)
    centroid_arr = np.array([centroid], dtype=np.float32)
    
    vec      = np.concatenate([ceps, zcr_arr, centroid_arr, f0_norm, sfm])
    norm     = np.linalg.norm(vec)
    return vec / (norm + 1e-9), sfm_val, f0_hz, zcr, centroid


_EMBEDDING_BACKEND: dict = {"fn": None}


def set_embedding_backend(fn: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
    _EMBEDDING_BACKEND["fn"] = fn
    logger.info(f"🔧 Embedding backend set to: {fn.__name__ if fn else 'built-in'}")


def _compute_embedding(pcm_bytes: bytes, sample_rate: int = _SAMPLE_RATE) -> tuple:
    custom_fn = _EMBEDDING_BACKEND.get("fn")
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    sfm_val = _spectral_flatness(pcm)
    f0_hz = _estimate_f0(pcm, sample_rate=sample_rate)
    zcr = _zero_crossing_rate(pcm)
    centroid = _spectral_centroid(pcm, sample_rate) / (sample_rate / 2)
    
    if custom_fn is not None:
        try:
            vec = np.asarray(custom_fn(pcm), dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-9), sfm_val, f0_hz, zcr, centroid
        except Exception as e:
            logger.error(f"❌ Custom embedding failed ({e}), using fallback")
            
    lfe   = _log_filterbank_energy(pcm, n_filters=N_FILTERS)
    ceps  = _cepstral_coeffs(lfe, n_ceps=N_CEPS)
    zcr_arr      = np.array([zcr], dtype=np.float32)
    centroid_arr = np.array([centroid], dtype=np.float32)
    f0_norm  = np.array([min(max(f0_hz, 0.0), _F0_MAX_HZ) / _F0_MAX_HZ], dtype=np.float32)
    sfm      = np.array([sfm_val], dtype=np.float32)
    vec      = np.concatenate([ceps, zcr_arr, centroid_arr, f0_norm, sfm])
    norm     = np.linalg.norm(vec)
    return vec / (norm + 1e-9), sfm_val, f0_hz, zcr, centroid


def _rms(pcm_bytes: bytes) -> float:
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(pcm) == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm ** 2)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _is_voice_band(sfm_val: float) -> bool:
    return _SFM_MUSIC_CEILING <= sfm_val <= _SFM_NOISE_FLOOR


def _compute_snr_db(rms_val: float, noise_floor: float) -> float:
    if noise_floor <= 0:
        return 30.0
    return 20.0 * _math.log10(max(rms_val, 1.0) / max(noise_floor, 1.0))


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
        self._profile_zcr_median: float = 0.0
        self._profile_zcr_mad: float = 0.01
        self._profile_centroid_median: float = 0.0
        self._profile_centroid_mad: float = 0.01
        
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

        self._barge_sustain_ms = 0.0
        self._contamination_detected = False

        self._audio_buffer: list[np.ndarray] = []
        self._buffer_chunk_count = 0
        self._similarity_ema: float | None = None
        self._recent_sims: collections.deque = collections.deque(maxlen=20)
        self._current_snr_db = 20.0
        self._enroll_quality_scores: list[float] = []
        
        self._last_decision: bool | None = None
        self._last_decision_time: float = 0.0

    def set_bot_speaking_ref(self, ref: dict) -> None:
        self._bot_speaking_ref = ref

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

    def _get_adaptive_threshold(self) -> float:
        threshold = self._cosine_threshold
        if _SNR_ADAPTIVE_ENABLED and self._state == LockState.LOCKED:
            snr = self._current_snr_db
            if snr < _SNR_BAD_DB:
                threshold -= _SNR_THRESHOLD_RELAX
            elif snr < _SNR_GOOD_DB:
                relax_factor = (_SNR_GOOD_DB - snr) / (_SNR_GOOD_DB - _SNR_BAD_DB)
                threshold -= _SNR_THRESHOLD_RELAX * relax_factor
            threshold = max(threshold, _ADAPT_MIN)
        return threshold

    def _update_similarity_ema(self, raw_similarity: float) -> float:
        if self._similarity_ema is None:
            self._similarity_ema = raw_similarity
        else:
            self._similarity_ema = (
                _SIMILARITY_SMOOTH_ALPHA * raw_similarity +
                (1 - _SIMILARITY_SMOOTH_ALPHA) * self._similarity_ema
            )
        return self._similarity_ema

    def _detect_overlap(self) -> bool:
        if len(self._recent_sims) < 5:
            return False
        recent = list(self._recent_sims)[-10:]
        variance = float(np.var(recent))
        return variance > 0.02

    def note_utterance_result(self, accepted: bool) -> None:
        if accepted:
            self._consecutive_utt_rejects = 0
            return
        self._consecutive_utt_rejects += 1
        if (
            self._state == LockState.LOCKED
            and self._fail_open_uses >= _FAIL_OPEN_MAX_USES
            and self._consecutive_utt_rejects >= _REENROLL_AFTER_REJECTS
        ):
            self.request_reenrollment(
                f"{self._consecutive_utt_rejects} consecutive rejects"
            )

    def request_reenrollment(self, reason: str = "") -> None:
        self._reenroll_count += 1
        logger.warning(f"🔄 RE-ENROLLMENT #{self._reenroll_count} ({reason})")
        self._state                   = LockState.ENROLLING
        self._profile                 = None
        self._profile_f0_median       = None
        self._profile_f0_mad          = 40.0
        self._profile_zcr_median      = 0.0
        self._profile_zcr_mad         = 0.01
        self._profile_centroid_median = 0.0
        self._profile_centroid_mad    = 0.01
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
        self._contamination_detected  = False
        self._audio_buffer            = []
        self._buffer_chunk_count      = 0
        self._similarity_ema          = None
        self._recent_sims.clear()
        self._enroll_quality_scores   = []
        self._last_decision = None
        self._last_decision_time = 0.0
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
            f"🟡 FAIL-OPEN #{self._fail_open_uses}/{_FAIL_OPEN_MAX_USES}: "
            f"{old:.2f} -> {self._cosine_threshold:.2f}"
        )
        if self._fail_open_uses >= _FAIL_OPEN_MAX_USES:
            self._pass_through = True

    def process_chunk(self, pcm_bytes: bytes, chunk_ms: float) -> tuple[bool, bool]:
        self._total += 1
        
        if self._bot_speaking_ref.get("bot_speaking"):
            return self._process_chunk_during_bot_speech(pcm_bytes, chunk_ms)

        rms_val = _rms(pcm_bytes)
        self._update_noise_floor(rms_val)
        self._current_snr_db = _compute_snr_db(rms_val, self._bg_rms_ema)

        if rms_val < _RMS_FLOOR_RATIO * self._bg_rms_ema:
            self._silence += 1
            self._last_decision = False 
            return True, True

        if self._state == LockState.ENROLLING:
            return self._enroll_chunk(pcm_bytes, chunk_ms, rms_val)

        if self._pass_through:
            self._accepted += 1
            return True, False

        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        self._audio_buffer.append(pcm_array.copy())
        self._buffer_chunk_count += 1

        if self._buffer_chunk_count < _EMBEDDING_BUFFER_CHUNKS:
            if self._last_decision is None:
                self._accepted += 1
                self._last_decision = True
                return True, False
            
            if self._last_decision:
                self._accepted += 1
            else:
                self._rejected += 1
            return self._last_decision, False

        buffered_audio = np.concatenate(self._audio_buffer)
        buffered_bytes = buffered_audio.tobytes()
        
        self._audio_buffer = []
        self._buffer_chunk_count = 0

        embedding, chunk_sfm, chunk_f0, chunk_zcr, chunk_centroid = _compute_embedding(buffered_bytes, self._sample_rate)

        raw_similarity = _cosine_similarity(embedding, self._profile)
        self._recent_sims.append(raw_similarity)
        smoothed_sim = self._update_similarity_ema(raw_similarity)

        current_time = time.monotonic()

        if (raw_similarity > _FAST_PATH_THRESHOLD and 
            self._last_decision is not None and 
            current_time - self._last_decision_time < 0.5):
            accepted = self._last_decision
            if accepted: self._accepted += 1
            else: self._rejected += 1
            return accepted, False

        if chunk_sfm < _SFM_MUSIC_CEILING or chunk_sfm > _SFM_NOISE_FLOOR:
            self._rejected += 1
            self._nonvoice_rejects += 1
            self._last_decision = False
            self._last_decision_time = current_time
            return False, False

        if self._looks_like_media(chunk_sfm):
            self._rejected += 1
            self._music_rejects += 1
            self._last_decision = False
            self._last_decision_time = current_time
            return False, False

        if raw_similarity < _HARD_REJECT_COSINE:
            self._rejected += 1
            self._last_decision = False
            self._last_decision_time = current_time
            return False, False

        self._utt_sims.append(raw_similarity)
        if len(self._utt_sims) > 400: del self._utt_sims[:200]
        if chunk_f0 > 0:
            self._utt_f0s.append(chunk_f0)
            if len(self._utt_f0s) > 400: del self._utt_f0s[:200]

        self._observe_similarity(raw_similarity)

        pitch_bonus = 0.0
        if chunk_f0 > 0 and self._profile_f0_median is not None:
            pitch_diff = abs(chunk_f0 - self._profile_f0_median)
            pitch_spread = max(self._profile_f0_mad, 25.0)
            if pitch_diff < pitch_spread: pitch_bonus = 0.05
            elif pitch_diff > pitch_spread * 3: pitch_bonus = -0.05

        is_overlap = self._detect_overlap()
        overlap_bonus = 0.0
        if is_overlap and rms_val > 1.5 * self._bg_rms_ema:
            overlap_bonus = 0.03

        effective_score = smoothed_sim + pitch_bonus + overlap_bonus
        threshold = self._get_adaptive_threshold()
        accepted = effective_score >= threshold

        # ==========================================
        # 🆘 STRICT OVERLAP RESCUE (NO LOUDNESS HIJACKING)
        # ==========================================
        if not accepted:
            is_borderline = raw_similarity > _HARD_REJECT_COSINE
            
            if is_borderline:
                rescue_score = 0.0
                
                # 1. Context Check
                recent_accepts = sum(1 for d in list(self._recent_sims)[-3:] if d > self._cosine_threshold)
                if recent_accepts >= 2:
                    rescue_score += 0.35
                
                # 🔥 FIX 2: Volume ONLY counts if identity is already somewhat matching
                # This prevents loud intruders from hijacking the rescue
                is_loud = rms_val > (1.8 * self._bg_rms_ema)
                if is_loud and raw_similarity > (self._cosine_threshold - 0.10):
                    rescue_score += 0.15
                
                # 3. Pitch (F0) Dominance Check
                if chunk_f0 > 0 and self._profile_f0_median is not None:
                    tol = max(40.0, self._profile_f0_median * 0.25)
                    if abs(chunk_f0 - self._profile_f0_median) < tol:
                        rescue_score += 0.30
                    elif abs(chunk_f0 - self._profile_f0_median) < tol * 1.5:
                        rescue_score += 0.15

                # 4. Spectral Shape Check
                zcr_match = abs(chunk_zcr - self._profile_zcr_median) < (2.5 * self._profile_zcr_mad)
                centroid_match = abs(chunk_centroid - self._profile_centroid_median) < (2.5 * self._profile_centroid_mad)
                
                if zcr_match and centroid_match:
                    rescue_score += 0.20
                elif zcr_match or centroid_match:
                    rescue_score += 0.10

                if _SEPARATION_ENABLED:
                    rescue_score += 0.05

                if rescue_score >= 0.65:
                    accepted = True
                    logger.info(f"🆘 OVERLAP RESCUE: score={rescue_score:.2f} (sim={raw_similarity:.2f})")

        self._last_decision = accepted
        self._last_decision_time = current_time

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
            self._echo_skipped += 1
            self._note_echo(rms_val)
            self._barge_sustain_ms = 0.0
            return False, False

        embedding, chunk_sfm, chunk_f0, chunk_zcr, chunk_centroid = _compute_embedding(pcm_bytes, self._sample_rate)
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
        
        # 🆘 STRICT BOT-SPEECH RESCUE
        if not accepted and raw_similarity > _HARD_REJECT_COSINE:
            rescue_score = 0.0
            
            if chunk_f0 > 0 and self._profile_f0_median is not None:
                tol = max(30.0, self._profile_f0_median * 0.15)
                if abs(chunk_f0 - self._profile_f0_median) < tol:
                    rescue_score += 0.45
                elif abs(chunk_f0 - self._profile_f0_median) < tol * 1.5:
                    rescue_score += 0.20
            
            zcr_match = abs(chunk_zcr - self._profile_zcr_median) < (2.0 * self._profile_zcr_mad)
            centroid_match = abs(chunk_centroid - self._profile_centroid_median) < (2.0 * self._profile_centroid_mad)
            
            if zcr_match and centroid_match:
                rescue_score += 0.35
            elif zcr_match or centroid_match:
                rescue_score += 0.15
                
            # 🔥 FIX 3: Volume only helps if identity is already very close
            if rms_val > (2.5 * self._bg_rms_ema) and raw_similarity > (self._cosine_threshold - 0.08):
                rescue_score += 0.20
                
            if rescue_score >= 0.75 and raw_similarity >= (self._cosine_threshold - 0.05):
                accepted = True
                self._barge_frames += 1
                self._barge_sustain_ms = 0.0
                logger.info(f"🆘 STRICT BOT RESCUE: score={rescue_score:.2f} (sim={raw_similarity:.2f})")

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
                                   self._base_threshold, _ADAPT_MAX)
            else:
                self._high_accept_counter = 0

        if abs(proposed - self._cosine_threshold) < 0.005:
            return
        old = self._cosine_threshold
        self._cosine_threshold = proposed
        self._adapted = True
        logger.debug(f"🎚️ Threshold: {old:.2f} -> {proposed:.2f}")

    def _check_auto_degrade(self) -> None:
        if self._pass_through or self._state != LockState.LOCKED:
            return
        decided = self._accepted + self._rejected
        if decided < _AUTO_DEGRADE_AFTER:
            return
        if self._rejected / decided <= _AUTO_DEGRADE_RATIO:
            return
        self._pass_through = True
        logger.error(f"🚨 AUTO-DEGRADED: {100.0*self._rejected/decided:.0f}% rejected")

    def _maybe_update_profile(self, embedding: np.ndarray, effective: float) -> None:
        if not _PROFILE_EMA_ENABLED or self._profile is None:
            return
        if effective < self._cosine_threshold + _PROFILE_EMA_MARGIN:
            return
        updated = (1.0 - _PROFILE_EMA_ALPHA) * self._profile + _PROFILE_EMA_ALPHA * embedding
        n = np.linalg.norm(updated)
        if n > 1e-6:
            self._profile = (updated / n).astype(np.float32)

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

    def _enroll_chunk(self, pcm_bytes: bytes, chunk_ms: float, rms_val: float):
        self._enroll_wall_ms += chunk_ms
        _, chunk_sfm, chunk_f0, chunk_zcr, chunk_centroid = _compute_embedding(pcm_bytes, self._sample_rate)

        if (
            not self._enroll_gate_relaxed
            and self._enroll_wall_ms >= (self._enroll_max_seconds * 1000.0) / 2.0
            and self._enrolled_ms < (self._enroll_seconds * 1000.0) / 2.0
        ):
            self._enroll_gate_relaxed = True
            self._enroll_gate_ratio = _ENV_ENROLL_RELAXED_SNR_RATIO
            logger.warning(f"⚠️ Enrollment starving, relaxing gate")

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
                    logger.warning(f"⚠️ Enrollment timeout with {self._enrolled_ms:.0f}ms")
            self._build_profile()

        self._accepted += 1
        return True, True

    def _build_profile(self):
        self._contamination_detected = False

        if not self._enroll_frames:
            logger.warning("⚠️ No qualified enroll frames")
            return

        combined = np.concatenate(self._enroll_frames)

        try:
            from app.embedding_backend import compute_embedding_quality
            quality = compute_embedding_quality(combined, self._sample_rate)
            self._enroll_quality_scores.append(quality)
            logger.info(f"📊 Enrollment quality score: {quality:.3f}")

            if quality < _MIN_ENROLL_QUALITY:
                logger.warning(f"⚠️ Low enrollment quality ({quality:.3f})")
        except ImportError:
            pass

        win = max(1600, _PROFILE_WINDOW_SAMPLES)
        vecs = []
        zcrs = []
        centroids = []
        for start in range(0, max(1, len(combined) - win + 1), win // 2 or win):
            chunk = combined[start:start + win]
            if len(chunk) < win // 2:
                continue
            v, sfm, f0, zcr, centroid = _compute_embedding(chunk.tobytes(), self._sample_rate)
            if _is_voice_band(sfm):
                vecs.append(v)
                zcrs.append(zcr)
                centroids.append(centroid)

        if len(vecs) >= 3:
            stacked = np.vstack(vecs)

            sims_matrix = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sim = float(np.dot(vecs[i], vecs[j]))
                    sims_matrix.append(sim)

            if sims_matrix:
                mean_inter_sim = float(np.mean(sims_matrix))
                std_inter_sim = float(np.std(sims_matrix))

                if mean_inter_sim < 0.60:
                    self._contamination_detected = True
                    logger.warning(
                        f"⚠️ CONTAMINATION DETECTED: "
                        f"inter-window similarity={mean_inter_sim:.3f} "
                        f"(expected >0.60)"
                    )

            profile = np.median(stacked, axis=0)
            n = np.linalg.norm(profile)
            self._profile = (profile / (n + 1e-9)).astype(np.float32)
            sims = [float(np.dot(v, self._profile)) for v in vecs]
            spread = float(np.std(sims))
            logger.info(
                f"🧬 Profile: {len(vecs)} windows, "
                f"mean={np.mean(sims):.3f} sd={spread:.3f}"
            )
            if spread > 0.12:
                self._low_confidence_profile = True
                self._cosine_threshold = max(
                    _ADAPT_MIN,
                    self._cosine_threshold - _LOW_CONF_THRESHOLD_DROP
                )
        else:
            self._profile, _, _, _, _ = _compute_embedding(combined.tobytes(), self._sample_rate)

        if self._enroll_f0_samples:
            arr = np.array(self._enroll_f0_samples, dtype=np.float32)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median))) * 1.4826
            self._profile_f0_median = median
            self._profile_f0_mad = max(mad, 25.0)
        else:
            self._profile_f0_median = None

        if zcrs:
            zcrs_arr = np.array(zcrs, dtype=np.float32) 
            self._profile_zcr_median = float(np.median(zcrs_arr))
            self._profile_zcr_mad = max(float(np.median(np.abs(zcrs_arr - self._profile_zcr_median))) * 1.4826, 0.005)
        else:
            self._profile_zcr_median = 0.0
            self._profile_zcr_mad = 0.01
            
        if centroids:
            centroids_arr = np.array(centroids, dtype=np.float32) 
            self._profile_centroid_median = float(np.median(centroids_arr))
            self._profile_centroid_mad = max(float(np.median(np.abs(centroids_arr - self._profile_centroid_median))) * 1.4826, 0.005)
        else:
            self._profile_centroid_median = 0.0
            self._profile_centroid_mad = 0.01

        self._state = LockState.LOCKED
        self._enroll_frames = []
        self._enroll_fallback_frames = []
        self._enroll_fallback_f0 = []
        logger.info(
            f"🔒 LOCKED after {self._enrolled_ms:.0f}ms | "
            f"threshold={self._cosine_threshold:.2f} | "
            f"contamination={self._contamination_detected}"
        )
        if self._frame_filter_ref is not None:
            self._frame_filter_ref.flush_history()

    def stats_str(self) -> str:
        total = self._accepted + self._rejected
        rej_pct = (100.0 * self._rejected / total) if total else 0.0
        return (
            f"SpeakerLock[{self._state.value}/{SPEAKER_LOCK_MODE}] "
            f"acc={self._accepted} rej={self._rejected} ({rej_pct:.0f}%) "
            f"sil={self._silence} echo={self._echo_skipped} "
            f"barge={self._barge_frames} "
            f"thr={self._cosine_threshold:.2f} "
            f"snr={self._current_snr_db:.1f}dB "
            f"buf={_EMBEDDING_BUFFER_MS}ms "
            f"contam={self._contamination_detected}"
        )


# ======================================================================
# SPEAKER-AWARE FRAME FILTER
# ======================================================================
class SpeakerAwareFrameFilter:
    def __init__(self, speaker_lock: SpeakerLock, history_window_ms: float = VAD_WINDOW_MS):
        self._lock = speaker_lock
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
        self._last_lock_decision = accepted
        now = time.monotonic()

        if (
            self._last_nonneutral_ts is not None
            and (now - self._last_nonneutral_ts) > _BURST_GAP_S
            and self._burst
        ):
            self._last_burst_ratio = sum(1 for ok in self._burst if ok) / len(self._burst)
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
        now = time.monotonic()
        cutoff = now - self._history_window_s
        recent = [(ts, ok) for ts, ok in self._decision_history if ts >= cutoff]
        if not recent:
            if self._last_lock_decision is False:
                return 0.0
            return 1.0
        return sum(1 for _, ok in recent if ok) / len(recent)

    def turn_evidence_chunks(self) -> int:
        now = time.monotonic()
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
        return is_caller, ratio, adjusted_mean, frames


# ======================================================================
# SPEAKER-GATED VAD
# ======================================================================
class SpeakerGatedVAD:
    def __init__(self, real_vad: Any, frame_filter: "SpeakerAwareFrameFilter",
                 bot_speaking_ref: dict = None):
        object.__setattr__(self, "_real_vad", real_vad)
        object.__setattr__(self, "_frame_filter", frame_filter)
        object.__setattr__(self, "_bot_speaking_ref", bot_speaking_ref or {"bot_speaking": False})
        object.__setattr__(self, "_suppressed_count", 0)
        object.__setattr__(self, "_passed_count", 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real_vad"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_real_vad", "_frame_filter", "_bot_speaking_ref",
                    "_suppressed_count", "_passed_count"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real_vad"), name, value)

    def stream(self, *args, **kwargs):
        real_vad = object.__getattribute__(self, "_real_vad")
        frame_filter = object.__getattribute__(self, "_frame_filter")
        bot_speaking_ref = object.__getattribute__(self, "_bot_speaking_ref")
        real_stream = real_vad.stream(*args, **kwargs)
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

    def __init__(self, real_stream: Any, frame_filter: SpeakerAwareFrameFilter,
                 gated_vad: SpeakerGatedVAD, bot_speaking_ref: dict):
        self._real_stream = real_stream
        self._frame_filter = frame_filter
        self._gated_vad = gated_vad
        self._bot_speaking_ref = bot_speaking_ref
        self._suppressing_current_utterance: bool = False

    def _get_event_type_name(self, event: Any) -> str | None:
        ev_type = getattr(event, "type", None)
        if ev_type is None:
            return None
        return (getattr(ev_type, "name", None) or getattr(ev_type, "value", None) or str(ev_type))

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
        ratio = self._frame_filter.turn_accept_ratio()
        bot_speaking = self._bot_speaking_ref.get("bot_speaking")
        threshold = VAD_ACCEPT_RATIO_BOT_SPEAKING if bot_speaking else VAD_ACCEPT_RATIO
        if ratio >= threshold:
            return False
        is_caller, _, mean_sim, frames = self._frame_filter.caller_verdict()
        if not is_caller:
            lock.note_utterance_result(False)
            return True
        if lock.fail_open_pending():
            lock.consume_fail_open()
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
        self._lock = speaker_lock
        self._pending_lang: str | None = None
        self._pending_count = 0

    def should_switch(self, new_lang: str, current_lang: str,
                      is_explicit: bool, chunk_was_verified: bool) -> Tuple[bool, str]:
        if is_explicit and chunk_was_verified:
            self.reset()
            return True, "explicit + verified"
        if is_explicit and not chunk_was_verified:
            self.reset()
            return True, "explicit (unverified)"
        if self._lock.state == LockState.ENROLLING:
            if self._pending_lang != new_lang:
                self._pending_lang = new_lang
                self._pending_count = 1
            else:
                self._pending_count += 1
            if self._pending_count >= _LANG_SWITCH_MIN_TURNS_ENROLLING:
                self.reset()
                return True, f"switch to '{new_lang}' confirmed"
            return False, f"switch pending"
        if not chunk_was_verified:
            self._reset_pending_if_different(new_lang)
            return False, "rejected by SpeakerLock"
        if self._pending_lang != new_lang:
            self._pending_lang = new_lang
            self._pending_count = 1
        else:
            self._pending_count += 1
        if self._pending_count >= _LANG_SWITCH_MIN_VERIFIED_TURNS:
            self.reset()
            return True, f"verified switch"
        return False, f"ambiguous switch pending"

    def reset(self):
        self._pending_lang = None
        self._pending_count = 0

    def _reset_pending_if_different(self, lang: str):
        if self._pending_lang and self._pending_lang != lang:
            self._pending_lang = None
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
# MAIN AUDIO PROCESSOR
# ======================================================================
class SpeakerLockAudioProcessor(rtc.FrameProcessor):
    def __init__(self, debug_record: bool = False, bot_speaking_ref: dict | None = None):
        super().__init__()
        self._debug = debug_record
        self._frame_rate = 8000
        self._raw_frames = []
        self._enabled = True
        self._bot_speaking_ref = bot_speaking_ref or {"bot_speaking": False}
        self._speaker_lock = SpeakerLock(bot_speaking_ref=self._bot_speaking_ref)
        self.frame_filter = SpeakerAwareFrameFilter(self._speaker_lock)
        self._noise_reducer = SpectralNoiseReducer() if _NOISE_REDUCTION_ENABLED else None
        self._rate_logged = False

        self._sep_enabled = _SEPARATION_ENABLED and _SEP_MODEL is not None
        if self._sep_enabled:
            self._audio_buffer = np.array([], dtype=np.int16)
            self._buffer_lock = threading.Lock()
            self._sep_queue = queue.Queue()
            self._sep_thread = threading.Thread(target=self._run_separation_worker, daemon=True)
            self._sep_thread.start()
            self._separated_audio = np.array([], dtype=np.int16)

        logger.info(f"🎤 SpeakerLockAudioProcessor ready (buffer={_EMBEDDING_BUFFER_MS}ms ULTRA FAST)")

    def _run_separation_worker(self):
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
                        self._separated_audio = chunk
                    else:
                        self._separated_audio = separated_streams[best_idx]
            except Exception as e:
                logger.error(f"❌ Separation error: {e}")
                self._separated_audio = chunk

    def _score_stream_full(self, pcm_int16: np.ndarray) -> float:
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
        emb, sfm, _, _, _ = _compute_embedding(chunk_bytes, self._frame_rate)
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
            logger.info(f"🎚️ First frame: {frame.sample_rate}Hz")

        raw_int16 = np.frombuffer(frame.data, dtype=np.int16)

        if self._sep_enabled and self._frame_rate == 8000:
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
            except Exception:
                pass
        logger.info(f"📊 {self._speaker_lock.stats_str()}")
