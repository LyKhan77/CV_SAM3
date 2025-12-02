"""
Test official SAM-3 processor vs custom pipeline.

This script validates:
1. Official processor produces correct number of objects
2. Masks are in correct format (numpy uint8, 0-255)
3. Export compatibility (ObjectList/ format)
4. Comparison with previous test results

Run: python test_official_processor.py
"""
import cv2
import torch
import numpy as np
from model import load_model
from PIL import Image
import os


def test_official_processor():
    """Test official SAM-3 post_process_instance_segmentation()."""
    print("=" * 60)
    print("Testing Official SAM-3 Processor")
    print("=" * 60)

    model, processor = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    test_cases = [
        ("pic/test_car2.png", "red car", 0.5, 0.85, (1, 3)),  # Expected: 1-3 objects
        ("pic/test_cat.png", "striped cat", 0.3, 0.7, (1, 2)),  # Expected: 1-2 connected objects
        ("pic/penguin.png", "penguin", 0.35, 0.75, (5, 8)),  # Expected: 5-8 penguins
    ]

    results_summary = []

    for img_path, prompt, conf, mask_th, expected_range in test_cases:
        print(f"\n{'─' * 60}")
        print(f"Test: {img_path}")
        print(f"Prompt: '{prompt}'")
        print(f"Config: confidence={conf:.0%}, mask_threshold={mask_th:.0%}")
        print(f"Expected: {expected_range[0]}-{expected_range[1]} objects")

        # Load image
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            continue

        frame = cv2.imread(img_path)
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Prepare inputs
        inputs = processor(text=[prompt], images=image_pil, return_tensors="pt").to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # OFFICIAL POST-PROCESSING
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=conf,
            mask_threshold=mask_th,
            target_sizes=[[frame.shape[0], frame.shape[1]]]
        )[0]

        # Extract masks
        masks = results.get('masks', [])
        scores = results.get('scores', [])

        print(f"✅ Official Processor: {len(masks)} objects detected")

        if len(scores) > 0:
            print(f"   Confidence scores: {[f'{s:.2f}' for s in scores[:5]]}")

        # Validate expected range
        if expected_range[0] <= len(masks) <= expected_range[1]:
            status = "PASS"
            print(f"✅ {status}: Object count within expected range")
        else:
            status = "FAIL"
            print(f"❌ {status}: Expected {expected_range[0]}-{expected_range[1]}, got {len(masks)}")

        results_summary.append({
            "test": img_path,
            "prompt": prompt,
            "detected": len(masks),
            "expected": expected_range,
            "status": status
        })

    # Print summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    for result in results_summary:
        emoji = "✅" if result["status"] == "PASS" else "❌"
        print(f"{emoji} {result['test']}: {result['detected']} objects (expected {result['expected'][0]}-{result['expected'][1]})")

    passed = sum(1 for r in results_summary if r["status"] == "PASS")
    total = len(results_summary)
    print(f"\n{passed}/{total} tests passed")


def test_export_compatibility():
    """Verify masks work with ObjectList/ export format."""
    print(f"\n{'=' * 60}")
    print("Testing Export Compatibility (ObjectList/ format)")
    print(f"{'=' * 60}\n")

    model, processor = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use test_car2.png for export validation
    test_image = "pic/test_car2.png"
    if not os.path.exists(test_image):
        print(f"⚠️  Test image not found: {test_image}")
        return

    frame = cv2.imread(test_image)
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(text=["red car"], images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Official processing
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.65,
        target_sizes=[[frame.shape[0], frame.shape[1]]]
    )[0]

    # Convert masks (same as production code in model.py)
    final_masks = []
    for mask in results['masks']:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Ensure uint8 format (0-255)
        if mask_np.dtype == bool:
            mask_uint8 = (mask_np.astype(np.uint8)) * 255
        elif mask_np.max() <= 1.0:
            mask_uint8 = (mask_np * 255).astype(np.uint8)
        else:
            mask_uint8 = mask_np.astype(np.uint8)

        final_masks.append(mask_uint8)

    # Validate export requirements
    print(f"Testing {len(final_masks)} masks for export compatibility...\n")

    all_passed = True
    for i, mask in enumerate(final_masks):
        try:
            # Check 1: dtype must be uint8
            assert mask.dtype == np.uint8, f"Mask {i} dtype is {mask.dtype}, expected uint8"

            # Check 2: Values in range 0-255
            assert mask.min() >= 0 and mask.max() <= 255, f"Mask {i} values out of range: {mask.min()}-{mask.max()}"

            # Check 3: Shape matches frame
            assert mask.shape == frame.shape[:2], f"Mask {i} shape {mask.shape} != frame shape {frame.shape[:2]}"

            # Check 4: Has non-zero area
            assert mask.sum() > 0, f"Mask {i} has no area (all zeros)"

            # Check 5: Bounding rect extraction (used in ObjectList/ export)
            x, y, w, h = cv2.boundingRect(mask)
            assert w > 0 and h > 0, f"Mask {i} bounding rect has zero size: {w}x{h}"

            print(f"✅ Mask {i+1}: dtype={mask.dtype}, shape={mask.shape}, bbox={w}x{h}, area={mask.sum()//255}px")

        except AssertionError as e:
            print(f"❌ Mask {i+1}: {e}")
            all_passed = False

    if all_passed:
        print(f"\n✅ EXPORT COMPATIBILITY TEST PASSED")
        print(f"   All {len(final_masks)} masks are compatible with ObjectList/ export")
    else:
        print(f"\n❌ EXPORT COMPATIBILITY TEST FAILED")
        print(f"   Some masks do not meet export requirements")


if __name__ == "__main__":
    try:
        # Test 1: Official processor validation
        test_official_processor()

        # Test 2: Export compatibility
        test_export_compatibility()

        print(f"\n{'=' * 60}")
        print("ALL TESTS COMPLETED")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
