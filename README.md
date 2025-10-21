# DIS (Densely-connected-Inception-Segmenter) Inference

This document provides instructions on how to run the background removal inference script (`Inference.py`).

## How to Run Inference

### 1. Place Your Input Images

Place the JPG images you want to process into the following directory:

```
demo_datasets/your_dataset/
```

By default, the script is configured to run on a sample image. You can either replace the images in that directory with your own, or you can modify the `Inference.py` script to point to your image file or directory.

To change the input path, open `IS-Net/Inference.py` and modify the `dataset_path` variable:

```python
# In IS-Net/Inference.py
dataset_path="../demo_datasets/your_dataset/your_image.jpg"  # Your dataset path
```

### 2. Execute the Script

Navigate to the `IS-Net` directory and run the inference script:

```bash
cd IS-Net
python Inference.py
```

### 3. Find the Output

The script will process the input image(s) and save the results as PNG files with the background removed. The output files will be located in:

```
demo_datasets/your_dataset_result/test/
```

The output files will have the same name as the input files but with a `.png` extension.