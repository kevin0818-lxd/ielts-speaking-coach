"""
UniSep-Inspired Pronunciation Scoring Model for IELTS Speaking.

Applies the core technique from "UniSep: Universal Target Audio Separation
with Language Models at Scale" (Wang et al., ICME 2025):
  - Audio tokenization via neural codec (EnCodec / RVQ)
  - Transformer sequence modeling in discrete latent space

Instead of audio separation, this model predicts pronunciation quality
from discrete audio tokens, trained on IELTS examiner scores.

Architecture:
  Audio -> EnCodec -> RVQ tokens (flattened) -> Transformer Encoder -> Scoring Heads
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. Codec Tokenizer  (wraps Facebook EnCodec)
# ---------------------------------------------------------------------------

class CodecTokenizer:
    """
    Converts raw audio waveforms to discrete token sequences using EnCodec.
    Follows UniSep Section II-A: flatten RVQ layers along the codebook dimension.
    """

    def __init__(self, bandwidth: float = 1.5, device: str = "cpu"):
        import encodec
        self.device = device
        self.model = encodec.EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(device).eval()
        self.sample_rate = self.model.sample_rate  # 24000
        self.codebook_size = self.model.quantizer.bins  # 1024
        # At 1.5kbps: 2 codebooks, 75 frames/sec
        self.n_codebooks = 2

    @torch.no_grad()
    def encode(self, waveform_16k: np.ndarray) -> torch.LongTensor:
        """
        Encode a 16kHz mono waveform to a flat sequence of codec tokens.

        Args:
            waveform_16k: numpy array, shape (samples,), 16kHz mono

        Returns:
            tokens: LongTensor, shape (seq_len,)
                    Flattened RVQ codes with codebook offsets applied so that
                    codebook-0 uses ids [0, 1023] and codebook-1 uses [1024, 2047].
        """
        import torchaudio

        wav = torch.from_numpy(waveform_16k).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)

        # Resample 16kHz -> 24kHz (EnCodec native rate)
        wav_24k = torchaudio.functional.resample(wav, 16000, self.sample_rate)
        wav_24k = wav_24k.to(self.device)

        encoded = self.model.encode(wav_24k)
        # encoded is a list of (codes, scale) tuples; codes shape: (B, n_q, T)
        codes = encoded[0][0]  # (1, n_q, T)
        codes = codes.squeeze(0)  # (n_q, T)

        # Take only first n_codebooks (at 1.5kbps bandwidth)
        codes = codes[: self.n_codebooks]  # (n_cb, T)

        # Apply codebook offsets: cb_i tokens are offset by i * codebook_size
        # This maps all codebooks into a single vocabulary space
        for i in range(codes.shape[0]):
            codes[i] += i * self.codebook_size

        # Flatten: interleave codebooks at each time step (following UniSep)
        # (n_cb, T) -> (T, n_cb) -> (T * n_cb,)
        tokens = codes.permute(1, 0).reshape(-1).cpu()
        return tokens

    @property
    def vocab_size(self) -> int:
        return self.codebook_size * self.n_codebooks  # 2048


# ---------------------------------------------------------------------------
# 2. Scoring Transformer  (UniSep-inspired sequence model)
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = 3  # [PAD]=0, [MASK]=1, [CLS]=2
PAD_ID = 0
MASK_ID = 1
CLS_ID = 2


class UniSepScoringTransformer(nn.Module):
    """
    Transformer encoder that operates on discrete audio tokens from EnCodec
    and predicts IELTS pronunciation / fluency / overall scores.

    Design choices following UniSep:
      - Discrete token embeddings (not continuous spectrogram)
      - Sinusoidal positional encoding
      - Causal-capable attention (used in pre-training; bidirectional in scoring)
      - Multi-task scoring heads
    """

    def __init__(
        self,
        vocab_size: int = 2048 + SPECIAL_TOKENS,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        max_seq_len: int = 4000,  # ~13s at 150 tokens/s * 2 codebooks
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Attention-weighted pooling for classification
        self.pool_attention = nn.Linear(d_model, 1)

        # Pre-training head: next-token prediction
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Scoring heads (1-9 IELTS scale)
        self.head_pronunciation = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.head_fluency = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.head_overall = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _attention_pool(self, hidden: torch.Tensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        """Attention-weighted mean pooling, ignoring padded positions."""
        # hidden: (B, T, D), padding_mask: (B, T) True=padded
        attn_logits = self.pool_attention(hidden).squeeze(-1)  # (B, T)
        attn_logits = attn_logits.masked_fill(padding_mask, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (hidden * attn_weights).sum(dim=1)  # (B, D)
        return pooled

    def forward_encoder(self, token_ids: torch.LongTensor, padding_mask: torch.BoolTensor = None):
        """
        Run the transformer encoder on token ids.

        Args:
            token_ids: (B, T) discrete audio tokens
            padding_mask: (B, T) True where padded

        Returns:
            hidden: (B, T, D) transformer output
        """
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        hidden = self.transformer(x, src_key_padding_mask=padding_mask)
        return hidden

    def forward_pretrain(self, token_ids: torch.LongTensor, padding_mask: torch.BoolTensor = None):
        """
        Pre-training forward pass: predict masked / next tokens.
        Returns logits over vocabulary at every position.
        """
        hidden = self.forward_encoder(token_ids, padding_mask)
        logits = self.lm_head(hidden)  # (B, T, vocab_size)
        return logits

    def forward(self, token_ids: torch.LongTensor, padding_mask: torch.BoolTensor = None):
        """
        Scoring forward pass.

        Returns:
            dict with keys: pronunciation, fluency, overall — each (B, 1)
        """
        hidden = self.forward_encoder(token_ids, padding_mask)
        pooled = self._attention_pool(hidden, padding_mask if padding_mask is not None else torch.zeros_like(token_ids, dtype=torch.bool))

        return {
            "pronunciation": self.head_pronunciation(pooled),
            "fluency": self.head_fluency(pooled),
            "overall": self.head_overall(pooled),
        }


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3. Model loading utility
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "resources", "unisep_scorer.pth"
)


def load_unisep_model(path: str = None, device: str = "cpu"):
    """
    Load the UniSep scoring transformer with trained weights.
    Returns (model, tokenizer) or (None, None) if weights are missing.
    """
    if path is None:
        path = _DEFAULT_WEIGHTS_PATH

    model = UniSepScoringTransformer()

    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state, strict=False)
            model.eval()
            print(f"[UniSep] Loaded scoring model from {path}")
        except Exception as e:
            print(f"[UniSep] Failed to load weights: {e}")
            return None, None
    else:
        print(f"[UniSep] No weights at {path} — model not available (run train_unisep_scorer.py first)")
        return None, None

    model.to(device)

    try:
        tokenizer = CodecTokenizer(device=device)
    except Exception as e:
        print(f"[UniSep] Failed to load EnCodec tokenizer: {e}")
        return None, None

    return model, tokenizer


def score_audio(model: UniSepScoringTransformer, tokenizer: CodecTokenizer,
                waveform_16k: np.ndarray, device: str = "cpu") -> dict:
    """
    Score a 16kHz mono waveform. Returns dict with pronunciation, fluency, overall (float).
    """
    tokens = tokenizer.encode(waveform_16k)

    # Truncate to max_seq_len
    if len(tokens) > model.max_seq_len:
        tokens = tokens[: model.max_seq_len]

    token_ids = tokens.unsqueeze(0).to(device)  # (1, T)
    # Offset by SPECIAL_TOKENS so codec ids don't collide with PAD/MASK/CLS
    token_ids = token_ids + SPECIAL_TOKENS

    padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        preds = model(token_ids, padding_mask)

    # Clamp to IELTS band range
    return {
        "pronunciation": max(1.0, min(9.0, preds["pronunciation"].item())),
        "fluency": max(1.0, min(9.0, preds["fluency"].item())),
        "overall": max(1.0, min(9.0, preds["overall"].item())),
    }
