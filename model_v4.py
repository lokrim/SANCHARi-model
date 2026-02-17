
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model_v4(encoder_name="tu-efficientnet_b4", encoder_weights="imagenet", in_channels=3, classes=1):
    """
    Creates the V4 Road Segmentation Model.
    
    Architecture: U-Net++ (Nested U-Net)
    Encoder: EfficientNet-B4 (via timm-universal)
    
    Why V4?
    - U-Net++: Dense skip connections improve gradient flow and capture fine-grained details better than U-Net.
    - EfficientNet-B4: Much stronger feature extractor than ResNet34, with better parameter efficiency.
    """
    # Create U-Net++ model
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,        
        encoder_weights=encoder_weights,  
        in_channels=in_channels,          
        classes=classes,  
        activation=None, # Return raw logits for numerical stability with BCEWithLogits/Focal
    )
    return model

if __name__ == '__main__':
    # Smoke Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = create_model_v4().to(device)
        # Test with 512x512 input (V4 standard)
        dummy_input = torch.randn(2, 3, 512, 512).to(device)
        output = model(dummy_input)
        print("V4 Model (EfficientNet-B4 + U-Net++) Instantiated Successfully")
        print(f"Output shape: {output.shape}")
        assert output.shape == (2, 1, 512, 512)
    except Exception as e:
        print(f"V4 Model Validation Failed: {e}")
        print("Ensure 'timm' is installed: pip install timm")
