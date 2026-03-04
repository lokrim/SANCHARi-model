
import torch
import segmentation_models_pytorch as smp

def create_model():
    """
    Creates a U-Net model with a pre-trained ResNet34 encoder.

    This model leverages transfer learning from ImageNet to improve feature extraction.
    
    Returns:
        A PyTorch model instance.
    """
    model = smp.Unet(
        encoder_name="resnet34",        # Use the ResNet34 architecture as the encoder
        encoder_weights="imagenet",      # Load weights pre-trained on ImageNet
        in_channels=3,                   # Number of input channels (RGB)
        classes=1,                       # Number of output classes (binary mask)
    )
    return model

if __name__ == '__main__':
    # A quick test to verify the model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the model
    model = create_model().to(device)
    
    # Create a dummy input tensor
    # Batch size = 2, Channels = 3, Height = 256, Width = 256
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    
    # Perform a forward pass
    output = model(dummy_input)
    
    # Print model summary and output shape
    print("SMP U-Net Model Instantiated Successfully")
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check if output shape is as expected
    # For binary segmentation, output should be [Batch_size, n_classes, H, W]
    assert output.shape == (2, 1, 256, 256)
    print("\nTest passed: Output shape is correct.")

