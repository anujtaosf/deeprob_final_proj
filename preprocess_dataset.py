"""
Animal Emotion Dataset Preprocessing Pipeline
==============================================
Removes backgrounds from animal photos so your CNN learns
facial/body features instead of grass, couches, etc.

HOW TO RUN:
    Basic:      python preprocess_dataset.py --input_dir ./data/raw --output_dir ./data/processed
    With viz:   python preprocess_dataset.py --input_dir ./data/raw --output_dir ./data/processed --visualize
    Save masks: python preprocess_dataset.py --input_dir ./data/raw --output_dir ./data/processed --save_masks

YOUR FOLDER STRUCTURE MUST LOOK LIKE THIS before running:
    data/raw/
        happy/
            dog1.jpg
            dog2.jpg
        sad/
            dog3.jpg
        anxious/
            dog4.jpg

OUTPUT will mirror that structure:
    data/processed/
        happy/
            dog1.png    <-- background removed, resized to 224x224
            dog2.png
        sad/
            dog3.png
        ...
"""

import argparse
from pathlib import Path      # pathlib lets us work with file paths cleanly (cross-platform)

import numpy as np
from PIL import Image         # Pillow: Python's main image manipulation library
from tqdm import tqdm         # tqdm: draws a progress bar in the terminal

# rembg is a background removal library built on top of a neural net called ISNet/U2-Net.
# new_session() loads the model weights into memory.
# remove() runs one image through the model and returns it with the background cut out.
from rembg import remove, new_session


# ==============================================================================
# STEP 1 — Background Removal
# ==============================================================================

def remove_background_rembg(img: Image.Image, session) -> tuple:
    """
    Runs the ISNet segmentation model on one image to cut out the background.

    HOW IT WORKS:
        ISNet is a deep neural net trained to predict which pixels belong to the
        "main subject" (foreground) vs. background. It outputs a probability map,
        which rembg thresholds into a binary alpha channel.

    WHAT IS AN ALPHA CHANNEL?
        Images normally have 3 channels: Red, Green, Blue (RGB).
        An RGBA image has a 4th channel: Alpha, which controls transparency.
        Alpha = 255  ->  pixel is fully visible (foreground / the animal)
        Alpha = 0    ->  pixel is fully transparent (background)
        rembg sets the background pixels' alpha to 0, making them see-through.

    INPUTS:
        img     -- a PIL Image already converted to RGBA mode
        session -- the loaded ISNet model session (created once in process_dataset,
                   then reused for every image so we don't reload weights each time)

    OUTPUTS:
        output  -- PIL RGBA Image: animal visible, background transparent
        mask    -- 2D numpy array shape (H, W), dtype uint8
                   Values: 255 = foreground pixel, 0 = background pixel
    """
    # Pass the image through the neural net. rembg handles all the inference internally.
    output = remove(img, session=session)

    # Extract just the alpha channel (index 3 of RGBA) as a numpy array.
    # This gives us a grayscale "mask" image we can inspect or save separately.
    mask = np.array(output)[:, :, 3]   # shape: (height, width), values 0-255

    return output, mask


# ==============================================================================
# STEP 2a — Background Replacement (choose white, black, or keep transparent)
# ==============================================================================

def apply_white_background(rgba: Image.Image) -> Image.Image:
    """
    Replaces the transparent background with solid white.

    WHY WHITE?
        Most pretrained CNNs (ResNet, EfficientNet, etc.) were trained on
        ImageNet photos which have varied, often bright backgrounds.
        White is a neutral choice that doesn't introduce new color information
        into the background region.

    HOW IT WORKS:
        We create a brand new solid-white RGB canvas the same size as our image,
        then "paste" the animal on top using the alpha channel as a stencil.
        Pixels where alpha=255 (the animal) get pasted; pixels where alpha=0
        (old background) stay white.

    INPUT:  RGBA PIL Image (transparent background)
    OUTPUT: RGB PIL Image  (white background)
    """
    background = Image.new("RGB", rgba.size, (255, 255, 255))  # pure white canvas
    background.paste(rgba, mask=rgba.split()[3])               # rgba.split()[3] = alpha channel used as mask
    return background


def apply_black_background(rgba: Image.Image) -> Image.Image:
    """
    Same as above but with a solid black background (0, 0, 0).
    Useful if you want dark-background augmentation.

    INPUT:  RGBA PIL Image
    OUTPUT: RGB PIL Image (black background)
    """
    background = Image.new("RGB", rgba.size, (0, 0, 0))
    background.paste(rgba, mask=rgba.split()[3])
    return background


# ==============================================================================
# STEP 2b — Crop to Subject (optional but recommended)
# ==============================================================================

def crop_to_subject(rgba: Image.Image, padding: int = 20) -> Image.Image:
    """
    Finds the bounding box of the animal and crops tightly around it.

    WHY CROP?
        Even after background removal, if your image is 1920x1080 and the dog
        is a small region in the corner, when you resize to 224x224 most of
        those pixels are just blank white. Cropping first means the resize step
        fills the frame with the actual animal, giving the model more detail.

    HOW IT WORKS:
        1. Look at the alpha channel mask (non-zero = foreground pixel).
        2. Find the first and last ROW that contains any foreground pixel -> top/bottom edges.
        3. Find the first and last COLUMN that contains any foreground pixel -> left/right edges.
        4. That gives us a tight bounding box. We expand it by `padding` pixels on each side
           so we don't accidentally clip fur/ear edges.
        5. Crop the RGBA image to that box.

    INPUTS:
        rgba    -- RGBA PIL Image with transparent background
        padding -- extra pixels to leave around the bounding box (default 20)

    OUTPUT: RGBA PIL Image, tightly cropped around the animal
    """
    mask = np.array(rgba)[:, :, 3]   # alpha channel, shape (H, W)

    # np.any(..., axis=1) asks: "does this ROW have ANY non-zero pixel?"
    # Result is a boolean array of length H (one True/False per row)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any():
        # No foreground found at all -- return the image unchanged to avoid crash
        return rgba

    # np.where(rows)[0] gives the indices of all True rows.
    # [[0, -1]] grabs the first and last index = top and bottom edge of the animal.
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Get image dimensions so we don't crop outside the image boundaries
    h, w = mask.shape

    # Expand the box by `padding` pixels, clamped to image edges
    rmin = max(0, rmin - padding)
    rmax = min(h, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w, cmax + padding)

    # PIL's crop() takes (left, upper, right, lower)
    return rgba.crop((cmin, rmin, cmax, rmax))


# ==============================================================================
# STEP 3 — process_image: orchestrates steps 1-2 for ONE image
# ==============================================================================

def process_image(
    img_path,              # full path to the input image file
    out_path,              # where to save the processed result (same relative path, different root)
    session,               # the already-loaded ISNet model (passed in so it isn't reloaded each call)
    background="white",    # what to replace the background with: "white", "black", or "transparent"
    crop=True,             # whether to crop tightly around the animal
    target_size=(224, 224),# final output resolution -- 224x224 is standard for ResNet/EfficientNet
    save_mask=False,       # if True, also save the binary alpha mask as a separate PNG
):
    """
    Runs the full preprocessing pipeline on a SINGLE image file.

    Pipeline for one image:
        img_path  ->  open & convert to RGBA
                  ->  remove_background_rembg()  ->  RGBA image + mask
                  ->  (optional) crop_to_subject()
                  ->  apply_*_background()        ->  RGB image
                  ->  resize to target_size
                  ->  save as PNG to out_path

    RETURNS:
        True  if the image was processed and saved successfully
        False if the file couldn't be opened (corrupted, wrong format, etc.)
    """

    # --- Open the image ---
    # Convert to RGBA immediately so every image has the same 4-channel format
    # regardless of whether it was originally JPG (RGB), PNG (RGBA), etc.
    try:
        img = Image.open(img_path).convert("RGBA")
    except Exception as e:
        print(f"  [SKIP] Could not open {img_path.name}: {e}")
        return False

    # --- Run background removal (Step 1) ---
    # rgba: the animal with a transparent background
    # mask: a 2D array where 255 = animal pixel, 0 = background pixel
    rgba, mask = remove_background_rembg(img, session)

    # --- Sanity check: did the model find an animal? ---
    # foreground_fraction = what fraction of the total pixels were labeled "animal"
    # If it's less than 1%, something probably went wrong (tiny subject or bad image)
    foreground_fraction = (mask > 128).sum() / mask.size
    if foreground_fraction < 0.01:
        print(f"  [WARN] Very small foreground in {img_path.name} ({foreground_fraction:.2%}) -- check this image")

    # --- Crop around the animal (Step 2b, optional) ---
    if crop:
        rgba = crop_to_subject(rgba, padding=20)

    # --- Replace background with chosen color (Step 2a) ---
    # We need an RGB image in the end because PyTorch's standard image loaders
    # expect 3-channel images. "transparent" keeps RGBA for special use cases.
    if background == "white":
        result = apply_white_background(rgba)
    elif background == "black":
        result = apply_black_background(rgba)
    elif background == "transparent":
        result = rgba          # stays as RGBA -- only use if your dataloader handles 4 channels
    else:
        result = apply_white_background(rgba)   # default fallback

    # --- Resize to target_size ---
    # Image.LANCZOS is a high-quality downsampling filter (similar to bicubic but sharper).
    # 224x224 is the standard input size for ResNet/EfficientNet trained on ImageNet.
    result = result.resize(target_size, Image.LANCZOS)

    # --- Save the processed image ---
    # Always save as PNG (not JPG) because JPG compression creates visible artifacts
    # around the animal-background edge where the mask transition is sharp.
    out_path.parent.mkdir(parents=True, exist_ok=True)    # create output subdirectory if needed
    save_path = out_path.with_suffix(".png")              # force .png extension
    result.save(save_path)

    # --- Optionally save the binary mask ---
    # The mask is useful if you later want to do masked-attention training
    # or to visually verify that segmentation is working correctly.
    if save_mask:
        mask_img = Image.fromarray(mask).resize(target_size, Image.NEAREST)
        # NEAREST interpolation for the mask so we don't blur the 0/255 values
        mask_path = out_path.parent / "masks" / (out_path.stem + "_mask.png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(mask_path)

    return True


# ==============================================================================
# STEP 4 — process_dataset: loops over ALL images in the dataset folder
# ==============================================================================

def process_dataset(
    input_dir,
    output_dir,
    background="white",
    crop=True,
    target_size=(224, 224),
    save_masks=False,
    visualize=False,
):
    """
    Entry point that processes an entire dataset folder.

    Walks the input directory recursively, finds every image file,
    and calls process_image() on each one. The class label subfolder
    structure is preserved in the output directory.

    WHY LOAD THE SESSION ONCE HERE?
        Loading the ISNet model from disk takes a few seconds and uses ~500MB RAM.
        If we loaded it inside process_image(), we'd reload it for every single photo.
        Instead, we load it once here and pass the same session object to every call.
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # --- Load the segmentation model ONCE ---
    # "isnet-general-use" is better than the default "u2net" for animals --
    # it handles fur edges and complex silhouettes more accurately.
    # On first run this downloads ~170MB model weights from the internet.
    # On subsequent runs it loads from a local cache (~/.u2net/).
    print("Loading segmentation model (downloads ~170MB on first run)...")
    session = new_session("isnet-general-use")
    print("Model loaded.\n")

    # --- Find all image files in the dataset ---
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [
        p for p in input_path.rglob("*")   # rglob("*") = recursive search through all subfolders
        if p.suffix.lower() in image_extensions
    ]

    print(f"Found {len(all_images)} images in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Background: {background} | Crop to subject: {crop} | Target size: {target_size}\n")

    success, skipped = 0, 0

    # --- Process each image ---
    # tqdm wraps the list and draws a progress bar like:
    # Processing: 47%|████      | 47/100 [00:23<00:26]
    for img_path in tqdm(all_images, desc="Processing"):

        # Build the output path by replacing the input root with the output root.
        # Example: data/raw/happy/dog1.jpg  ->  data/processed/happy/dog1.png
        # This preserves the class label subfolder so PyTorch's ImageFolder can read it.
        rel = img_path.relative_to(input_path)   # e.g. "happy/dog1.jpg"
        out_path = output_path / rel              # e.g. "data/processed/happy/dog1.jpg" (suffix overwritten inside process_image)

        ok = process_image(
            img_path, out_path, session,
            background=background,
            crop=crop,
            target_size=target_size,
            save_mask=save_masks,
        )
        if ok:
            success += 1
        else:
            skipped += 1

    print(f"\nDone. {success} processed, {skipped} skipped.")

    # --- Optional: save a before/after visual comparison ---
    if visualize:
        visualize_samples(input_path, output_path, n=5)


# ==============================================================================
# BONUS -- visualize_samples: saves a before/after grid image for sanity checking
# ==============================================================================

def visualize_samples(input_path, output_path, n=5):
    """
    Saves a PNG grid showing 'Original' vs 'Processed' side by side for n images.
    Useful to quickly verify the segmentation is working before training.
    The file is saved as: <output_dir>/preprocessing_samples.png
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed -- run: pip install matplotlib")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    # Grab the first n images from the raw dataset as examples
    originals = [p for p in input_path.rglob("*") if p.suffix.lower() in image_extensions][:n]

    # Create a 2-row grid: row 0 = originals, row 1 = processed
    fig = plt.figure(figsize=(4 * n, 4))
    gs = gridspec.GridSpec(2, n)

    for i, orig_path in enumerate(originals):
        # Find the corresponding processed image (same relative path, .png extension)
        rel = orig_path.relative_to(input_path)
        processed_path = (output_path / rel).with_suffix(".png")

        ax_orig = fig.add_subplot(gs[0, i])
        ax_proc = fig.add_subplot(gs[1, i])

        ax_orig.imshow(Image.open(orig_path))
        ax_orig.set_title("Original", fontsize=8)
        ax_orig.axis("off")

        if processed_path.exists():
            ax_proc.imshow(Image.open(processed_path))
            ax_proc.set_title("Processed", fontsize=8)
        else:
            ax_proc.set_title("Not found", fontsize=8)
        ax_proc.axis("off")

    plt.tight_layout()
    viz_path = output_path / "preprocessing_samples.png"
    plt.savefig(viz_path, dpi=150)
    print(f"Saved visualization to {viz_path}")


# ==============================================================================
# COMMAND-LINE INTERFACE -- parses arguments when you run the script directly
# ==============================================================================

if __name__ == "__main__":
    # argparse reads flags you pass in the terminal and stores them in `args`
    parser = argparse.ArgumentParser(description="Preprocess animal emotion dataset")

    # Required: where your raw images live and where to write outputs
    parser.add_argument("--input_dir",  required=True, help="Path to raw dataset folder (must have class subfolders)")
    parser.add_argument("--output_dir", required=True, help="Where to save processed images")

    # Optional: controls for how images are processed
    parser.add_argument("--background", default="white", choices=["white", "black", "transparent"],
                        help="Color to fill where the background was (default: white)")
    parser.add_argument("--no_crop",    action="store_true",
                        help="Pass this flag to DISABLE cropping around the animal")
    parser.add_argument("--size",       type=int, default=224,
                        help="Output image size in pixels (square). Default 224 matches ResNet/EfficientNet input")
    parser.add_argument("--save_masks", action="store_true",
                        help="Also save the binary alpha masks as separate PNGs in a /masks subfolder")
    parser.add_argument("--visualize",  action="store_true",
                        help="Save a before/after comparison image to <output_dir>/preprocessing_samples.png")

    args = parser.parse_args()

    # Call the main pipeline function with the parsed arguments
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        background=args.background,
        crop=not args.no_crop,            # --no_crop flag flips this to False
        target_size=(args.size, args.size),
        save_masks=args.save_masks,
        visualize=args.visualize,
    )