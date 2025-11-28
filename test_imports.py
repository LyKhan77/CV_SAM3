#!/usr/bin/env python3
import sys
print("Python path:", sys.executable)
print("Testing imports...")

try:
    import uvicorn
    print("✓ uvicorn imported successfully")
except ImportError as e:
    print("✗ uvicorn import failed:", e)

try:
    import fastapi
    print("✓ fastapi imported successfully")
except ImportError as e:
    print("✗ fastapi import failed:", e)

try:
    import torch
    print("✓ torch imported successfully")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name()}")
except ImportError as e:
    print("✗ torch import failed:", e)

try:
    import transformers
    print("✓ transformers imported successfully")
except ImportError as e:
    print("✗ transformers import failed:", e)

try:
    from transformers import Sam3Processor, Sam3Model
    print("✓ Sam3Processor and Sam3Model imported successfully")
except ImportError as e:
    print("✗ Sam3 import failed:", e)

print("\nTrying to import our custom modules...")
try:
    import model
    print("✓ model.py imported successfully")
except ImportError as e:
    print("✗ model.py import failed:", e)

print("\nAll tests completed!")