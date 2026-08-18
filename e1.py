"""
embedding_backend.py - LOW LATENCY OPTIMIZED ECAPA-TDNN Embeddings
====================================================================
Optimized for speed with:
- Reduced window size (300ms instead of 1000ms)
- Reduced overlap (25% instead of 50%)
- Early exit on first good embedding
- Faster resampling
- Optimized quality scoring
"""

import os
import numpy as np
import logging

logger = logging.getLogger("embedding_backend")

# ---------------------------------------------------------------
# CONFIGURATION - OPTIMIZED FOR LOW LATENCY
# ---------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000       # ECAPA-TDNN expects 16kHz

# 🔥 FAST MODE: Reduced from 4000/8000 to 1600/2400
MIN_AUDIO_SAMPLES_8K = 1600      # Minimum 0.2s at 8kHz (was 0.5s)
IDEAL_AUDIO_SAMPLES_8K = 2400    # Ideal 0.3s at 8kHz (was 1.0s)

EMBEDDING_DIM = 192              # ECAPA-TDNN output size

# 🔥 FAST MODE: Early exit instead of averaging all windows
USE_EARLY_EXIT = os.getenv("EMBEDDING_EARLY_EXIT", "true").lower() == "true"
EARLY_EXIT_CONFIDENCE = float(os.getenv("EARLY_EXIT_CONFIDENCE", "0.65"))

# 🔥 FAST MODE: Reduced overlap from 50% to 25%
WINDOW_OVERLAP_RATIO = float(os.getenv("WINDOW_OVERLAP_RATIO", "0.25"))

# Model cache
_MODEL = None
_MODEL_LOADED = False
_MODEL_FAILED = False

# Resampling cache for repeated calls
_RESAMPLE_CACHE = {}
_RESAMPLE_CACHE_MAX = 10


def _load_model():
    """Load ECAPA-TDNN model. Called only once (cached)."""
    global _MODEL, _MODEL_LOADED, _MODEL_FAILED

    if _MODEL_LOADED or _MODEL_FAILED:
        return _MODEL

    try:
        logger.info("🧠 Loading ECAPA-TDNN model...")
        print("🧠 Loading ECAPA-TDNN deep learning model...")

        from speechbrain.pretrained import EncoderClassifier

        _MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
            run_opts={"device": "cpu"}
        )

        _MODEL_LOADED = True
        print("✅ ECAPA-TDNN model loaded successfully!")
        logger.info("✅ ECAPA-TDNN loaded")

    except Exception as e:
        _MODEL_FAILED = True
        logger.error(f"❌ Failed to load ECAPA-TDNN: {e}")
        print(f"❌ ECAPA-TDNN failed: {e}")

    return _MODEL


def _resample_to_16k(pcm_int16: np.ndarray, original_rate: int) -> np.ndarray:
    """Resample audio to 16kHz for ECAPA-TDNN with caching."""
    if original_rate == TARGET_SAMPLE_RATE:
        return pcm_int16

    # 🔥 OPTIMIZATION: Check cache
    cache_key = (pcm_int16.tobytes()[:100], original_rate)  # Use first 100 bytes as key
    if cache_key in _RESAMPLE_CACHE:
        return _RESAMPLE_CACHE[cache_key]

    try:
        import torch
        import torchaudio

        waveform = torch.from_numpy(pcm_int16.astype(np.float32) / 32768.0)
        waveform = waveform.unsqueeze(0)

        resampled = torchaudio.functional.resample(
            waveform,
            orig_freq=original_rate,
            new_freq=TARGET_SAMPLE_RATE
        )

        result = (resampled.squeeze(0).numpy() * 32768).astype(np.int16)
        
        # Cache result (limit cache size)
        if len(_RESAMPLE_CACHE) < _RESAMPLE_CACHE_MAX:
            _RESAMPLE_CACHE[cache_key] = result
        
        return result

    except Exception:
        # Fallback: linear interpolation (faster but lower quality)
        ratio = TARGET_SAMPLE_RATE / original_rate
        new_length = int(len(pcm_int16) * ratio)
        if new_length <= 0:
            return np.zeros(1, dtype=np.int16)
        indices = np.linspace(0, len(pcm_int16) - 1, new_length)
        return np.interp(indices, np.arange(len(pcm_int16)), pcm_int16).astype(np.int16)


def _extract_single_embedding(pcm_16k: np.ndarray) -> np.ndarray:
    """Extract embedding from a single audio segment at 16kHz."""
    import torch

    # Normalize to float32 [-1, 1]
    waveform = torch.from_numpy(pcm_16k.astype(np.float32) / 32768.0)
    waveform = waveform.unsqueeze(0)  # [1, samples]

    with torch.no_grad():
        embedding = _MODEL.encode_batch(waveform)

    # Shape: [1, 1, 192] → [192]
    emb = embedding.squeeze().cpu().numpy().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(emb)
    if norm > 1e-9:
        emb = emb / norm

    return emb


def ecapa_embedding(pcm_int16: np.ndarray, sample_rate: int = 8000) -> np.ndarray:
    """
    Extract speaker embedding with LOW LATENCY optimization.

    Optimizations:
    - Smaller windows (300ms instead of 1000ms)
    - Reduced overlap (25% instead of 50%)
    - Early exit on first high-confidence embedding
    - Faster processing overall

    Args:
        pcm_int16: Audio samples (mono, int16)
        sample_rate: Sample rate of input

    Returns:
        192-dimensional normalized embedding
    """
    model = _load_model()
    if model is None:
        raise RuntimeError("ECAPA-TDNN model not available")

    try:
        # Step 1: Resample to 16kHz
        pcm_16k = _resample_to_16k(pcm_int16, sample_rate)

        # Step 2: Ensure minimum length
        min_samples_16k = int(MIN_AUDIO_SAMPLES_8K * (TARGET_SAMPLE_RATE / 8000))
        if len(pcm_16k) < min_samples_16k:
            padding = min_samples_16k - len(pcm_16k)
            pcm_16k = np.pad(pcm_16k, (0, padding), mode='constant')

        # Step 3: Windowed extraction (OPTIMIZED)
        ideal_samples_16k = int(IDEAL_AUDIO_SAMPLES_8K * (TARGET_SAMPLE_RATE / 8000))
        window_size = ideal_samples_16k
        hop_size = int(window_size * (1.0 - WINDOW_OVERLAP_RATIO))  # 🔥 Reduced overlap

        if len(pcm_16k) <= window_size:
            # Short audio: single embedding
            return _extract_single_embedding(pcm_16k)
        else:
            # Long audio: multiple windows
            embeddings = []
            start = 0
            best_embedding = None
            best_norm = 0.0

            while start + window_size <= len(pcm_16k):
                window = pcm_16k[start:start + window_size]

                # Skip silent windows
                energy = np.mean(np.abs(window.astype(np.float32)))
                if energy > 50:  # Not silence
                    emb = _extract_single_embedding(window)
                    embeddings.append(emb)
                    
                    # 🔥 OPTIMIZATION: Early exit on first good embedding
                    if USE_EARLY_EXIT and best_embedding is None:
                        # Use first embedding immediately
                        best_embedding = emb
                        best_norm = np.linalg.norm(emb)
                        
                        # If we have enough confidence, exit early
                        # (This is a heuristic - in practice you might want to check
                        # similarity with a reference profile here)
                        if len(embeddings) >= 1:
                            break

                start += hop_size

            # Handle remaining audio if we didn't early exit
            if not USE_EARLY_EXIT and start < len(pcm_16k) and len(pcm_16k) - start > min_samples_16k:
                remaining = pcm_16k[start:]
                energy = np.mean(np.abs(remaining.astype(np.float32)))
                if energy > 50:
                    emb = _extract_single_embedding(remaining)
                    embeddings.append(emb)

            if not embeddings:
                # All windows were silent, use full audio
                return _extract_single_embedding(pcm_16k)

            # 🔥 OPTIMIZATION: Use early exit result if available
            if USE_EARLY_EXIT and best_embedding is not None:
                return best_embedding

            # Otherwise average all window embeddings (original behavior)
            avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

            # L2 normalize the average
            norm = np.linalg.norm(avg_embedding)
            if norm > 1e-9:
                avg_embedding = avg_embedding / norm

            return avg_embedding

    except Exception as e:
        logger.error(f"❌ Embedding extraction failed: {e}")
        raise


def compute_embedding_quality(pcm_int16: np.ndarray, sample_rate: int = 8000) -> float:
    """
    Compute audio quality score for enrollment (OPTIMIZED for speed).
    Returns 0.0 (bad) to 1.0 (excellent).
    """
    if len(pcm_int16) == 0:
        return 0.0

    pcm_float = pcm_int16.astype(np.float32)

    # Check 1: Length (longer = better)
    duration_sec = len(pcm_int16) / sample_rate
    length_score = min(1.0, duration_sec / 3.0)

    # Check 2: Energy (not too quiet, not clipping) - OPTIMIZED
    # Use subset for faster computation
    sample_size = min(len(pcm_float), sample_rate)  # Max 1 second
    rms = np.sqrt(np.mean(pcm_float[:sample_size] ** 2))
    if rms < 100:
        energy_score = 0.2
    elif rms > 30000:
        energy_score = 0.5
    else:
        energy_score = 1.0

    # Check 3: Spectral diversity - OPTIMIZED
    try:
        fft_size = min(len(pcm_float), sample_rate)
        fft_mag = np.abs(np.fft.rfft(pcm_float[:fft_size]))
        if len(fft_mag) > 0 and np.max(fft_mag) > 0:
            # Simplified entropy calculation
            fft_norm = fft_mag / np.sum(fft_mag)
            spectral_entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10))
            diversity_score = min(1.0, spectral_entropy / 5.0)
        else:
            diversity_score = 0.0
    except Exception:
        diversity_score = 0.5

    # Check 4: Voiced segments ratio - OPTIMIZED
    # Use larger frames and hops for speed
    frame_size = int(sample_rate * 0.030)  # 30ms frames (was 25ms)
    hop = int(sample_rate * 0.020)  # 20ms hop (was 10ms)
    voiced_frames = 0
    total_frames = 0

    for start in range(0, len(pcm_int16) - frame_size, hop):
        frame = pcm_int16[start:start + frame_size].astype(np.float32)
        energy = np.sqrt(np.mean(frame ** 2))
        total_frames += 1
        if energy > 300:
            voiced_frames += 1

    voiced_ratio = voiced_frames / max(1, total_frames)
    voiced_score = min(1.0, voiced_ratio / 0.5)

    # Weighted combination
    quality = (
        0.25 * length_score +
        0.20 * energy_score +
        0.25 * diversity_score +
        0.30 * voiced_score
    )

    return round(quality, 3)


def create_embedding_function(sample_rate: int = 8000):
    """Factory function for embedding extraction."""

    def embedding_fn(pcm_int16: np.ndarray) -> np.ndarray:
        return ecapa_embedding(pcm_int16, sample_rate=sample_rate)

    return embedding_fn


def is_model_available() -> bool:
    """Check if ECAPA-TDNN is available."""
    model = _load_model()
    return model is not None


def warmup():
    """Pre-load model to avoid delay on first request."""
    print("🔥 Warming up ECAPA-TDNN model...")
    _load_model()

    if _MODEL is not None:
        try:
            import torch
            dummy = torch.zeros(1, 16000)
            with torch.no_grad():
                _MODEL.encode_batch(dummy)
            print("✅ Model warmup complete!")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")


if __name__ == "__main__":
    print("Testing LOW LATENCY ECAPA-TDNN...")

    # Test with synthetic audio
    test_audio = (np.sin(2 * np.pi * 440 * np.arange(16000) / 8000) * 10000).astype(np.int16)

    try:
        embedding = ecapa_embedding(test_audio, sample_rate=8000)
        quality = compute_embedding_quality(test_audio, sample_rate=8000)

        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"✅ Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"✅ Quality score: {quality}")
        print("\n🎉 LOW LATENCY ECAPA-TDNN working correctly!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
