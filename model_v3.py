
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model_v3(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    """
    Creates the V3 Road Segmentation Model using a U-Net architecture.
    
    This function wraps `segmentation_models_pytorch` to provide a consistent
    interface for creating the model with pre-defined defaults suitable for
    road extraction tasks.

    Args:
        encoder_name (str): The backbone architecture (default: "resnet34").
                            ResNet34 offers a good balance between speed and accuracy.
        encoder_weights (str): Pre-trained weights to initialize the encoder (default: "imagenet").
                               Transfer learning significantly speeds up convergence.
        in_channels (int): Number of input channels (default: 3 for RGB).
        classes (int): Number of output classes (default: 1 for binary segmentation).

    Returns:
        torch.nn.Module: The configured PyTorch model ready for training or inference.
    """
    # Create U-Net model with specified encoder
    model = smp.Unet(
        encoder_name=encoder_name,        
        encoder_weights=encoder_weights,  
        in_channels=in_channels,          
        classes=classes,                  
    )
    return model

if __name__ == '__main__':
    # A quick test to verify the model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model_v3().to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    output = model(dummy_input)
    print("V3 Model (ResNet34-UNet) Instantiated Successfully")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 1, 256, 256)
