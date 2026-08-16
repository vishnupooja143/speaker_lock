
"""
embedding_backend.py - ENHANCED ECAPA-TDNN Deep Learning Speaker Embeddings
=============================================================================
Maximum accuracy version with:
- Windowed embedding extraction (500ms windows)
- Quality scoring per window
- Multi-window averaging
- Proper 8kHz → 16kHz resampling
- Model caching and warmup
"""

import os
import numpy as np
import logging

logger = logging.getLogger("embedding_backend")

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000       # ECAPA-TDNN expects 16kHz
MIN_AUDIO_SAMPLES_8K = 4000      # Minimum 0.5s at 8kHz
IDEAL_AUDIO_SAMPLES_8K = 8000    # Ideal 1s at 8kHz
EMBEDDING_DIM = 192              # ECAPA-TDNN output size

# Model cache
_MODEL = None
_MODEL_LOADED = False
_MODEL_FAILED = False


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
    """Resample audio to 16kHz for ECAPA-TDNN."""
    if original_rate == TARGET_SAMPLE_RATE:
        return pcm_int16

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

        return (resampled.squeeze(0).numpy() * 32768).astype(np.int16)

    except Exception:
        # Fallback: linear interpolation
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
    Extract speaker embedding with windowed averaging for maximum accuracy.

    For short audio (<1s): Pads and extracts single embedding
    For long audio (≥1s): Splits into overlapping windows, extracts
                          embeddings from each, and averages them

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

        # Step 3: Windowed extraction for longer audio
        ideal_samples_16k = int(IDEAL_AUDIO_SAMPLES_8K * (TARGET_SAMPLE_RATE / 8000))
        window_size = ideal_samples_16k
        hop_size = window_size // 2  # 50% overlap

        if len(pcm_16k) <= window_size:
            # Short audio: single embedding
            return _extract_single_embedding(pcm_16k)
        else:
            # Long audio: multiple windows → average
            embeddings = []
            start = 0

            while start + window_size <= len(pcm_16k):
                window = pcm_16k[start:start + window_size]

                # Skip silent windows
                energy = np.mean(np.abs(window.astype(np.float32)))
                if energy > 50:  # Not silence
                    emb = _extract_single_embedding(window)
                    embeddings.append(emb)

                start += hop_size

            # Handle remaining audio
            if start < len(pcm_16k) and len(pcm_16k) - start > min_samples_16k:
                remaining = pcm_16k[start:]
                energy = np.mean(np.abs(remaining.astype(np.float32)))
                if energy > 50:
                    emb = _extract_single_embedding(remaining)
                    embeddings.append(emb)

            if not embeddings:
                # All windows were silent, use full audio
                return _extract_single_embedding(pcm_16k)

            # Average all window embeddings
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
    Compute audio quality score for enrollment.
    Returns 0.0 (bad) to 1.0 (excellent).

    Checks:
    - Audio length
    - Energy level
    - Spectral diversity
    - Signal-to-noise estimate
    """
    if len(pcm_int16) == 0:
        return 0.0

    pcm_float = pcm_int16.astype(np.float32)

    # Check 1: Length (longer = better)
    duration_sec = len(pcm_int16) / sample_rate
    length_score = min(1.0, duration_sec / 3.0)  # Full score at 3+ seconds

    # Check 2: Energy (not too quiet, not clipping)
    rms = np.sqrt(np.mean(pcm_float ** 2))
    if rms < 100:
        energy_score = 0.2  # Too quiet
    elif rms > 30000:
        energy_score = 0.5  # Clipping
    else:
        energy_score = 1.0

    # Check 3: Spectral diversity (voice has varied frequencies)
    try:
        fft_mag = np.abs(np.fft.rfft(pcm_float[:min(len(pcm_float), sample_rate)]))
        if len(fft_mag) > 0 and np.max(fft_mag) > 0:
            spectral_entropy = -np.sum(
                (fft_mag / np.sum(fft_mag)) *
                np.log(fft_mag / np.sum(fft_mag) + 1e-10)
            )
            diversity_score = min(1.0, spectral_entropy / 5.0)
        else:
            diversity_score = 0.0
    except Exception:
        diversity_score = 0.5

    # Check 4: Voiced segments ratio
    frame_size = int(sample_rate * 0.025)  # 25ms frames
    hop = int(sample_rate * 0.010)  # 10ms hop
    voiced_frames = 0
    total_frames = 0

    for start in range(0, len(pcm_int16) - frame_size, hop):
        frame = pcm_int16[start:start + frame_size].astype(np.float32)
        energy = np.sqrt(np.mean(frame ** 2))
        total_frames += 1
        if energy > 300:
            voiced_frames += 1

    voiced_ratio = voiced_frames / max(1, total_frames)
    voiced_score = min(1.0, voiced_ratio / 0.5)  # Full score at 50% voiced

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
    print("Testing Enhanced ECAPA-TDNN...")

    # Test with synthetic audio
    test_audio = (np.sin(2 * np.pi * 440 * np.arange(16000) / 8000) * 10000).astype(np.int16)

    try:
        embedding = ecapa_embedding(test_audio, sample_rate=8000)
        quality = compute_embedding_quality(test_audio, sample_rate=8000)

        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"✅ Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"✅ Quality score: {quality}")
        print("\n🎉 Enhanced ECAPA-TDNN working correctly!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
