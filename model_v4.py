
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def create_model_v4(
    encoder_name="tu-efficientnet_b4",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
):
    """
    Constructs the V4 road segmentation model.

    Architecture:
        Decoder : U-Net++ (nested dense skip connections)
        Encoder : EfficientNet-B4 via the timm-universal hub

    Design rationale:
        - EfficientNet-B4 provides richer feature representations than ResNet34
          through compound scaling of depth, width, and resolution.
        - U-Net++ replaces standard single-level skip connections with dense
          nested sub-networks, reducing the semantic gap between encoder and
          decoder and improving fine-detail recovery for thin structures.
        - Raw logits are returned (no activation) for numerical stability when
          used with BCEWithLogitsLoss or FocalLoss.

    Args:
        encoder_name    (str): timm model identifier for the encoder backbone.
        encoder_weights (str): Pre-trained weight set ('imagenet' or None).
        in_channels     (int): Number of input image channels.
        classes         (int): Number of output segmentation classes.

    Returns:
        torch.nn.Module: Initialised U-Net++ model ready for training or inference.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


if __name__ == "__main__":
    # Smoke test: verify model instantiation and output shape on 512x512 input.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = create_model_v4().to(device)
        dummy_input = torch.randn(2, 3, 512, 512).to(device)
        output = model(dummy_input)
        print("V4 Model (EfficientNet-B4 + U-Net++) instantiated successfully.")
        print(f"Output shape: {output.shape}")
        assert output.shape == (2, 1, 512, 512)
    except Exception as e:
        print(f"V4 model validation failed: {e}")
        print("Ensure 'timm' is installed: pip install timm")
