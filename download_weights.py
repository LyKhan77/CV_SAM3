"""
Script to download SAM model weights
"""
import os
import urllib.request
from pathlib import Path

# Model options
MODELS = {
    'vit_h': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'filename': 'sam_vit_h_4b8939.pth',
        'size': '2.4 GB',
        'description': 'ViT-H - Best accuracy (Recommended)'
    },
    'vit_l': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'filename': 'sam_vit_l_0b3195.pth',
        'size': '1.2 GB',
        'description': 'ViT-L - Good balance'
    },
    'vit_b': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'filename': 'sam_vit_b_01ec64.pth',
        'size': '375 MB',
        'description': 'ViT-B - Fastest'
    }
}

def download_with_progress(url, destination):
    """Download file with progress bar"""
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        mb_downloaded = (count * block_size) / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f'\rDownloading... {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='')

    print(f'Downloading to: {destination}')
    urllib.request.urlretrieve(url, destination, reporthook)
    print('\nDownload complete!')

def main():
    # Create weights directory
    weights_dir = Path(__file__).parent / 'model' / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAM Model Weights Downloader")
    print("=" * 60)
    print("\nAvailable models:")
    for key, model in MODELS.items():
        print(f"\n{key.upper()}:")
        print(f"  - {model['description']}")
        print(f"  - Size: {model['size']}")
        print(f"  - File: {model['filename']}")

    print("\n" + "=" * 60)
    choice = input("\nChoose model (vit_h/vit_l/vit_b) [default: vit_h]: ").strip().lower()

    if not choice:
        choice = 'vit_h'

    if choice not in MODELS:
        print(f"Invalid choice. Using default: vit_h")
        choice = 'vit_h'

    model = MODELS[choice]
    destination = weights_dir / model['filename']

    # Check if file already exists
    if destination.exists():
        overwrite = input(f"\n{model['filename']} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Download cancelled.")
            return

    print(f"\nDownloading {choice.upper()} model ({model['size']})...")
    print(f"URL: {model['url']}")

    try:
        download_with_progress(model['url'], str(destination))
        print(f"\nModel saved to: {destination}")
        print("\nYou can now run the application with SAM support!")
        print("Run: python app.py")
    except Exception as e:
        print(f"\nError during download: {e}")
        print("\nAlternative: Download manually from:")
        print(f"  {model['url']}")
        print(f"\nThen save to:")
        print(f"  {destination}")

if __name__ == '__main__':
    main()
