from dataclasses import dataclass, field


export_2_onnx: bool = True
@dataclass
class MambaConfig:

    d_model: int = 2560
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    def __post_init__(self):
        self.fused_add_norm = self.fused_add_norm and not export_2_onnx
