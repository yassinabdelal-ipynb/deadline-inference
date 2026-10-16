"""Supported image classifiers and checkpoint identities, without importing torch.

The application measures image-classification inference. A local checkpoint must
contain parameters for one of these architectures; it cannot supply Python code.
"""

from __future__ import annotations

DEFAULT_MODEL = "mobilenet_v3_small"
MAX_IMPORTED_WEIGHT_BYTES = 256 * 1024 * 1024

_MODELS = {
    "mobilenet_v3_small": {
        "label": "MobileNetV3-Small",
        "weights_enum": "MobileNet_V3_Small_Weights",
        "filename": "mobilenet_v3_small-047dcff4.pth",
        "sha256": "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
        "max_weight_bytes": 32 * 1024 * 1024,
    },
    "resnet18": {
        "label": "ResNet18",
        "weights_enum": "ResNet18_Weights",
        "filename": "resnet18-f37072fd.pth",
        "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        "max_weight_bytes": 64 * 1024 * 1024,
    },
    "efficientnet_b0": {
        "label": "EfficientNet-B0",
        "weights_enum": "EfficientNet_B0_Weights",
        "filename": "efficientnet_b0_rwightman-7f5810bc.pth",
        "sha256": "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934",
        "max_weight_bytes": 32 * 1024 * 1024,
    },
    "squeezenet1_1": {
        "label": "SqueezeNet1.1",
        "weights_enum": "SqueezeNet1_1_Weights",
        "filename": "squeezenet1_1-b8a52dc0.pth",
        "sha256": "b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b",
        "max_weight_bytes": 16 * 1024 * 1024,
    },
}


def get_model(model_id: str = DEFAULT_MODEL) -> dict:
    """Return a JSON-compatible copy of one supported architecture's specification."""
    if not isinstance(model_id, str) or model_id not in _MODELS:
        raise ValueError("Unsupported image classifier. Choose: " + ", ".join(_MODELS))
    spec = dict(_MODELS[model_id])
    return {
        **spec,
        "id": model_id,
        "model_id": model_id,
        "task": "image classification",
        "weights_name": spec["weights_enum"] + ".IMAGENET1K_V1",
        "url": "https://download.pytorch.org/models/" + spec["filename"],
        "input_channels": 3,
        "default_image_size": 224,
        "default_num_classes": 1000,
    }


def list_models() -> list[dict]:
    """List supported architectures; this is safe to call before torch is loaded."""
    return [get_model(model_id) for model_id in _MODELS]


get_model_spec = get_model


def validate_selection(model_id=DEFAULT_MODEL, weights_path="", num_classes=1000, image_size=224):
    """Validate bounded image-only configuration without loading libraries/files."""
    get_model(model_id)
    if not isinstance(weights_path, str):
        raise ValueError("weights_path must be a local checkpoint path or an empty string")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or not 1 <= num_classes <= 10000:
        raise ValueError("num_classes must be an integer from 1 through 10000")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or not 64 <= image_size <= 512:
        raise ValueError("image_size must be an integer from 64 through 512 pixels")
    if not weights_path and num_classes != 1000:
        raise ValueError("Official pretrained weights have 1000 classes. Import matching custom weights for another class count.")
