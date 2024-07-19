# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    ImagePoolingAttn,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    ResNetLayer,
    ContrastiveHead,
    BNContrastiveHead,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
    Add,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
from .conv_lora import(
    Conv_lora,
    C2f_lora,
    SPPF_lora,
)
from .conv_lora_m import(
    Conv_lora_m,
    C2f_lora_m,
    SPPF_lora_m,
)
from .conv_all_lora import(
    Conv_all_lora,
    C2f_all_lora,
    SPPF_all_lora,
)
from .conv_all_lora_m import(
    Conv_all_lora_m,
    C2f_all_lora_m,
    SPPF_all_lora_m,
)
from .conv_adalora import(
    Conv_adalora,
    C2f_adalora,
    SPPF_adalora,
)
from .conv_adalora_symmetric_m import(
    Conv_adalora_sym_m,
    C2f_adalora_sym_m,
    SPPF_adalora_sym_m,
)
from .conv_adalora_asymmetric_m import(
    Conv_adalora_asym_m,
    C2f_adalora_asym_m,
    SPPF_adalora_asym_m,
)


__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "Conv_lora",
    "C2f_lora",
    "SPPF_lora",
    "Add",
    "Conv_lora_m",
    "C2f_lora_m",
    "SPPF_lora_m",
    "Conv_all_lora",
    "C2f_all_lora",
    "SPPF_all_lora",
    "Conv_all_lora_m",
    "C2f_all_lora_m",
    "SPPF_all_lora_m",
    "Conv_adalora",
    "C2f_adalora",
    "SPPF_adalora",
    "Conv_adalora_sym_m",
    "C2f_adalora_sym_m",
    "SPPF_adalora_sym_m",
    "Conv_adalora_asym_m",
    "C2f_adalora_asym_m",
    "SPPF_adalora_asym_m",
)
