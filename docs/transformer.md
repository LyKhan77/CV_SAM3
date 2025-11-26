# Gated model: Login with a HF token with gated access permission
hf auth login

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("mask-generation", model="facebook/sam3")     

# Load model directly
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/sam3")
model = AutoModel.from_pretrained("facebook/sam3")