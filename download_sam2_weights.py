"""
Download SAM 2 Model Checkpoints

This script downloads SAM 2 model weights from the official repository.
For 4GB VRAM, we recommend 'hiera_base_plus' model.
"""

import os
import urllib.request
from pathlib import Path

# SAM 2 checkpoint URLs (from official GitHub releases)
SAM2_CHECKPOINTS = {
    'hiera_tiny': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        'size': '~150MB',
        'vram': '~800MB'
    },
    'hiera_small': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
        'size': '~200MB',
        'vram': '~1.2GB'
    },
    'hiera_base_plus': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'size': '~350MB',
        'vram': '~1.5GB'
    },
    'hiera_large': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
        'size': '~900MB',
        'vram': '~3GB'
    }
}

def download_checkpoint(model_type='hiera_base_plus', output_dir='model/weights'):
    """
    Download SAM 2 checkpoint

    Args:
        model_type: One of 'hiera_tiny', 'hiera_small', 'hiera_base_plus', 'hiera_large'
        output_dir: Directory to save checkpoint
    """

    if model_type not in SAM2_CHECKPOINTS:
        print(f"Error: Invalid model type '{model_type}'")
        print(f"Available models: {list(SAM2_CHECKPOINTS.keys())}")
        return False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get checkpoint info
    checkpoint_info = SAM2_CHECKPOINTS[model_type]
    url = checkpoint_info['url']
    filename = f'sam2_hiera_{model_type.split("_")[1]}'
    if len(model_type.split('_')) > 2:
        filename += f'_{model_type.split("_")[2]}'
    filename += '.pt'

    output_file = output_path / filename

    # Check if already downloaded
    if output_file.exists():
        print(f"✓ Checkpoint already exists: {output_file}")
        print(f"  Size: {checkpoint_info['size']}")
        print(f"  VRAM: {checkpoint_info['vram']}")
        return True

    print(f"Downloading SAM 2 '{model_type}' checkpoint...")
    print(f"  URL: {url}")
    print(f"  Size: {checkpoint_info['size']}")
    print(f"  Output: {output_file}")

    try:
        # Download with progress
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 100 / totalsize
                s = f"\r  Progress: {percent:5.1f}% ({readsofar:,} / {totalsize:,} bytes)"
                print(s, end='', flush=True)

        urllib.request.urlretrieve(url, output_file, reporthook)
        print()  # New line after progress

        print(f"✓ Download complete: {output_file}")
        print(f"  VRAM requirement: {checkpoint_info['vram']}")
        return True

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if output_file.exists():
            output_file.unlink()  # Remove partial download
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("SAM 2 Checkpoint Downloader")
    print("=" * 60)
    print()
    print("Available models:")
    for model, info in SAM2_CHECKPOINTS.items():
        print(f"  - {model:20s} Size: {info['size']:10s} VRAM: {info['vram']}")
    print()

    # For 4GB VRAM, use hiera_base_plus
    recommended_model = 'hiera_base_plus'
    print(f"Recommended for 4GB VRAM: {recommended_model}")
    print()

    # Download
    success = download_checkpoint(recommended_model)

    if success:
        print()
        print("=" * 60)
        print("✓ Setup complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the application: python app.py")
    else:
        print()
        print("=" * 60)
        print("✗ Download failed")
        print("=" * 60)
        print()
        print("Manual download:")
        print(f"1. Visit: {SAM2_CHECKPOINTS[recommended_model]['url']}")
        print(f"2. Save to: model/weights/sam2_hiera_base_plus.pt")
