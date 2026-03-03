
"""
Sanchari V4 - Local Batch Inference Script (predict_v4.py)

Runs sliding-window inference with 4-way Test Time Augmentation (TTA) on a
directory of local image files and saves all intermediate and final outputs.

Output files per image:
    {name}_input.jpg    -- Original input image (BGR).
    {name}_prob.png     -- Raw probability map (8-bit greyscale).
    {name}_mask.png     -- Binary road mask after post-processing.
    {name}_skeleton.png -- 1-pixel-wide road centreline skeleton.
    {name}_overlay.jpg  -- Input image with mask overlaid in red.

Usage:
    python predict_v4.py
    python predict_v4.py --input test-images --output predictedv4
"""

import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
import rasterio

from model_v4 import create_model_v4
from postprocess_v4 import apply_advanced_postprocessing


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_IMAGES_DIR = "test-images"
OUTPUT_DIR      = "predicted/predictedv4"
MODEL_PATH      = "weights/best_model_v4.pth"
PATCH_SIZE      = 512   # Inference patch size (pixels).
STRIDE          = 256   # Sliding-window step: 50 % overlap.

# ImageNet normalisation statistics required by the EfficientNet-B4 encoder.
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_sliding_window(large_image, model, device):
    """
    Runs sliding-window inference with 4-way TTA on a large image.

    The image is padded with reflect-border so that every pixel is covered
    by at least one window.  Four prediction passes are averaged per patch:
    original, horizontal flip, vertical flip, and 90-degree rotation.

    Args:
        large_image (np.ndarray):    Input image (H, W, 3) in RGB, uint8.
        model       (nn.Module):     Trained segmentation model in eval mode.
        device      (torch.device):  Compute device.

    Returns:
        np.ndarray: Float32 probability map of shape (H, W) in [0, 1].
    """
    h, w, _ = large_image.shape
    prob_map  = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    padded = cv2.copyMakeBorder(large_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    h_pad, w_pad, _ = padded.shape

    mean = NORM_MEAN.to(device)
    std  = NORM_STD.to(device)

    for y in range(0, h_pad - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w_pad - PATCH_SIZE + 1, STRIDE):
            patch = padded[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            # Convert (H, W, C) -> (1, C, H, W), normalise, send to device.
            t = torch.from_numpy(patch.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            t = (t - mean) / std

            with torch.no_grad():
                probs     = torch.sigmoid(model(t)).squeeze().cpu().numpy()
                probs_h   = torch.flip(torch.sigmoid(model(torch.flip(t, [3]))), [3]).squeeze().cpu().numpy()
                probs_v   = torch.flip(torch.sigmoid(model(torch.flip(t, [2]))), [2]).squeeze().cpu().numpy()
                probs_rot = torch.rot90(torch.sigmoid(model(torch.rot90(t, 1, [2, 3]))), -1, [2, 3]).squeeze().cpu().numpy()

            probs_avg = (probs + probs_h + probs_v + probs_rot) / 4.0
            prob_map [y : y + PATCH_SIZE, x : x + PATCH_SIZE] += probs_avg
            count_map[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

    count_map[count_map == 0] = 1
    prob_map /= count_map
    return prob_map[:h, :w]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V4 batch inference on local images.")
    parser.add_argument("--input",  default=TEST_IMAGES_DIR, help="Directory of input images.")
    parser.add_argument("--output", default=OUTPUT_DIR,      help="Directory for output predictions.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model_v4().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"V4 weights loaded from {MODEL_PATH}.")
    else:
        print(f"Error: Weights not found at {MODEL_PATH}.")
        return
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    test_images = glob.glob(os.path.join(args.input, "*.jpg"))
    if not test_images:
        print(f"No images found in {args.input}.")
        return

    print(f"Found {len(test_images)} images. Starting inference ...")

    for img_path in tqdm(test_images):
        base_name = os.path.basename(img_path).split(".")[0]

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prob_map = predict_sliding_window(image_rgb, model, device)

        binary_mask, skeleton, _ = apply_advanced_postprocessing(prob_map, threshold=0.45)

        skeleton_uint8 = (skeleton     * 255).astype(np.uint8)
        mask_uint8     = (binary_mask  * 255).astype(np.uint8)
        prob_uint8     = (prob_map     * 255).astype(np.uint8)

        # Save all output artefacts.
        cv2.imwrite(os.path.join(args.output, f"{base_name}_input.jpg"),    image)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_prob.png"),     prob_uint8)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_mask.png"),     mask_uint8)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_skeleton.png"), skeleton_uint8)

        overlay = image.copy()
        overlay[mask_uint8 > 0] = [0, 0, 255]
        combined = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(args.output, f"{base_name}_overlay.jpg"), combined)

    print(f"Inference complete. Results saved to {args.output}.")


if __name__ == "__main__":
    main()
