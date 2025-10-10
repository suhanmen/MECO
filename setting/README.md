You may encounter errors with flash-attn==2.6.3 depending on your CUDA version.

If this happens, install it using:
"pip install flash-attn --extra-index-url https://download.pytorch.org/whl/cu??"
Replace the "??" at the end with your CUDA version (for example, use 126 for CUDA 12.6).
