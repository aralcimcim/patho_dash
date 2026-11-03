"""Model-specific official recipes; imports stay lazy for CPU-only export/tests."""

from dataclasses import dataclass

MODELS = {
    "resnet18": {
        "checkpoint": "torchvision/ResNet18_Weights.IMAGENET1K_V1",
        "dimension": 512,
        "pooling": "global average pool; fc=Identity",
        "source": "https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html",
    },
    "phikon-v2": {
        "checkpoint": "owkin/phikon-v2",
        "dimension": 1024,
        "pooling": "last_hidden_state[:, 0, :] (CLS)",
        "source": "https://huggingface.co/owkin/phikon-v2",
    },
    "uni2-h": {
        "checkpoint": "MahmoodLab/UNI2-h",
        "dimension": 1536,
        "pooling": "model output: CLS; exclude eight register tokens",
        "source": "https://huggingface.co/MahmoodLab/UNI2-h",
    },
}


class ModelUnavailable(RuntimeError):
    """Missing approval or authentication, never replaced with random weights."""


@dataclass
class Encoder:
    model: object
    transform: object
    spec: dict
    kind: str

    def forward(self, images):
        if self.kind == "phikon-v2":
            return self.model(pixel_values=images).last_hidden_state[:, 0, :]
        return self.model(images)


class ProcessorTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(image, return_tensors="pt")["pixel_values"][0]


def load_encoder(name, revision=None):
    import torch

    from .common import file_hash

    if name not in MODELS:
        raise ValueError(
            f"Unknown encoder {name}; implement a documented adapter before adding it."
        )
    spec = dict(MODELS[name])
    if name == "resnet18":
        from pathlib import Path

        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        transform = weights.transforms()
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
        spec.update(revision=file_hash(checkpoint), weights_url=weights.url)
        if not spec["revision"].startswith("f37072fd"):
            raise ValueError("Unexpected ResNet checkpoint hash.")
    else:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

        try:
            # Resolve main once, then use the immutable SHA for weights AND preprocessing.
            resolved = (
                HfApi().model_info(spec["checkpoint"], revision=revision or "main").sha
            )
            spec["revision"] = resolved
            if name == "phikon-v2":
                from transformers import AutoImageProcessor, AutoModel

                processor = AutoImageProcessor.from_pretrained(
                    spec["checkpoint"], revision=resolved, use_fast=False
                )
                transform = ProcessorTransform(processor)
                spec["processor"] = processor.to_dict()
                model = AutoModel.from_pretrained(spec["checkpoint"], revision=resolved)
            else:
                import timm
                from torchvision import transforms

                # Official UNI2-h local-checkpoint recipe. Strict loading must succeed.
                checkpoint = hf_hub_download(
                    spec["checkpoint"], "pytorch_model.bin", revision=resolved
                )
                model = timm.create_model(
                    "vit_giant_patch14_224",
                    pretrained=False,
                    img_size=224,
                    patch_size=14,
                    depth=24,
                    num_heads=24,
                    init_values=1e-5,
                    embed_dim=1536,
                    mlp_ratio=2.66667 * 2,
                    num_classes=0,
                    no_embed_class=True,
                    mlp_layer=timm.layers.SwiGLUPacked,
                    act_layer=torch.nn.SiLU,
                    reg_tokens=8,
                    dynamic_img_size=True,
                )
                model.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=True),
                    strict=True,
                )
                transform = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                        ),
                    ]
                )
        except (GatedRepoError, HfHubHTTPError, OSError) as exc:
            # from_pretrained wraps access errors in OSError; other loading errors propagate.
            if (
                isinstance(exc, GatedRepoError)
                or "401" in str(exc)
                or "403" in str(exc)
                or "gated" in str(exc).lower()
            ):
                raise ModelUnavailable(
                    f"{name}: request access at {spec['source']}, then supply HF_TOKEN."
                ) from exc
            raise
    spec["preprocessing"] = (
        repr(transform.processor)
        if isinstance(transform, ProcessorTransform)
        else repr(transform)
    )
    spec["input"] = (
        "RGB, source patches 224x224 at 0.5 MPP; resize does not change physical acquisition"
    )
    model.requires_grad_(False)
    model.eval()
    return Encoder(model, transform, spec, name)
