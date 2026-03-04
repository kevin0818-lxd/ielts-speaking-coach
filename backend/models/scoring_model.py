import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBranchScoringModel(nn.Module):
    """
    A deep learning model that fuses multiple feature dimensions for IELTS Speaking assessment.

    Architecture:
    1. Spectral Branch (MFCC): Processed by Bi-LSTM to capture temporal patterns.
    2. Prosodic Branch (F0, Energy): Processed by Lightweight LSTM.
    3. SSL Branch (wav2vec2 / HuBERT): Optional frozen self-supervised representations
       projected and attention-pooled for rich phonetic/fluency signal.
    4. Fusion Layer: Concatenates high-level representations + Static Features (WPM, Pauses).
    5. Multi-Task Heads: Predicts Pronunciation, Fluency, Lexical, Grammar, and Overall Score.
    """
    def __init__(self,
                 input_dim_mfcc=40,
                 hidden_dim_mfcc=64,
                 input_dim_prosody=2,
                 hidden_dim_prosody=16,
                 static_features_dim=10,
                 ssl_feature_dim=768,
                 ssl_proj_dim=64,
                 use_ssl=False,
                 dropout=0.3):
        super(MultiBranchScoringModel, self).__init__()

        self.use_ssl = use_ssl
        self.dropout_rate = dropout

        # --- Branch 1: Spectral (MFCC) ---
        self.spectral_lstm = nn.LSTM(
            input_size=input_dim_mfcc,
            hidden_size=hidden_dim_mfcc,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.spectral_attention = nn.Linear(hidden_dim_mfcc * 2, 1)
        self.spectral_dropout = nn.Dropout(dropout)

        # --- Branch 2: Prosodic (F0, Energy) ---
        self.prosody_lstm = nn.LSTM(
            input_size=input_dim_prosody,
            hidden_size=hidden_dim_prosody,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.prosody_dropout = nn.Dropout(dropout)

        # --- Branch 3: SSL (wav2vec2 / HuBERT) ---
        if use_ssl:
            self.ssl_proj = nn.Linear(ssl_feature_dim, ssl_proj_dim)
            self.ssl_attention = nn.Linear(ssl_proj_dim, 1)
            self.ssl_layer_norm = nn.LayerNorm(ssl_proj_dim)
            self.ssl_dropout = nn.Dropout(dropout)

        # --- Fusion Layer ---
        fusion_dim = (hidden_dim_mfcc * 2) + hidden_dim_prosody + static_features_dim
        if use_ssl:
            fusion_dim += ssl_proj_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Multi-Task Output Heads ---
        self.head_pronunciation = nn.Linear(32, 1)
        self.head_fluency = nn.Linear(32, 1)
        self.head_lexical = nn.Linear(32, 1)
        self.head_grammar = nn.Linear(32, 1)
        self.head_overall = nn.Linear(32 + 4, 1)

    def forward(self, mfcc, prosody, static_features, ssl_features=None):
        """
        Args:
            mfcc: (Batch, Time, 40)
            prosody: (Batch, Time, 2)
            static_features: (Batch, 10)
            ssl_features: (Batch, Time', ssl_feature_dim) — optional, only when use_ssl=True
        """
        # --- Branch 1: Spectral ---
        spectral_out, _ = self.spectral_lstm(mfcc)
        attn_weights = F.softmax(self.spectral_attention(spectral_out), dim=1)
        spectral_context = torch.sum(spectral_out * attn_weights, dim=1)
        spectral_context = self.spectral_dropout(spectral_context)

        # --- Branch 2: Prosodic ---
        _, (h_n, _) = self.prosody_lstm(prosody)
        prosody_context = h_n[-1]
        prosody_context = self.prosody_dropout(prosody_context)

        # --- Fusion parts ---
        parts = [spectral_context, prosody_context, static_features]

        # --- Branch 3: SSL ---
        if self.use_ssl and ssl_features is not None:
            ssl_proj = F.relu(self.ssl_proj(ssl_features))
            ssl_proj = self.ssl_layer_norm(ssl_proj)
            ssl_attn = F.softmax(self.ssl_attention(ssl_proj), dim=1)
            ssl_context = torch.sum(ssl_proj * ssl_attn, dim=1)
            ssl_context = self.ssl_dropout(ssl_context)
            parts.append(ssl_context)

        combined = torch.cat(parts, dim=1)
        shared_representation = self.fusion_fc(combined)

        # --- Multi-Task Prediction ---
        pron_score = self.head_pronunciation(shared_representation)
        fluency_score = self.head_fluency(shared_representation)
        lexical_score = self.head_lexical(shared_representation)
        grammar_score = self.head_grammar(shared_representation)

        overall_input = torch.cat((
            shared_representation,
            pron_score, fluency_score, lexical_score, grammar_score
        ), dim=1)

        overall_score = self.head_overall(overall_input)

        return {
            "pronunciation": pron_score,
            "fluency": fluency_score,
            "lexical": lexical_score,
            "grammar": grammar_score,
            "overall": overall_score
        }


def load_model(path=None, use_ssl=False, ssl_feature_dim=768, ssl_proj_dim=64):
    if path is None:
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "scoring_model.pth")

    model = MultiBranchScoringModel(
        use_ssl=use_ssl,
        ssl_feature_dim=ssl_feature_dim,
        ssl_proj_dim=ssl_proj_dim,
    )

    import os
    if os.path.exists(path):
        try:
            print(f"Loading DL model from {path}...")
            ckpt = torch.load(path, map_location='cpu')
            if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                ckpt = ckpt["state_dict"]

            model_state = model.state_dict()
            compatible = {}
            skipped = []

            for k, v in ckpt.items():
                if k not in model_state:
                    skipped.append((k, "missing_in_model"))
                    continue
                if hasattr(v, "shape") and hasattr(model_state[k], "shape") and v.shape != model_state[k].shape:
                    skipped.append((k, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}"))
                    continue
                compatible[k] = v

            model.load_state_dict(compatible, strict=False)
            model.eval()
            if skipped:
                preview = ", ".join([f"{k}({reason})" for k, reason in skipped[:5]])
                print(f"Loaded with partial compatibility: {len(compatible)} keys loaded, {len(skipped)} skipped. e.g. {preview}")
        except Exception as e:
            print(f"Failed to load model weights: {e}")
    else:
        print(f"No model weights found at {path}. Using untrained model.")

    return model
