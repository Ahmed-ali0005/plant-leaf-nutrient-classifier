# Plant Leaf Nutrient Deficiency Detection 

This project uses **deep learning with MobileNetV2** to detect **nutrient deficiencies in plants** from images of their leaves.  
By analyzing leaf photos, the model can identify whether a plant is healthy or suffering from a specific deficiency.

---

## Detected Classes
The model can classify a leaf into one of the following categories:

- **Healthy**
- Boron deficiency
- Calcium deficiency
- Copper deficiency
- Iron deficiency
- Magnesium deficiency
- Manganese deficiency
- Molybdenum deficiency
- Nitrogen deficiency
- Phosphorus deficiency
- Potassium deficiency
- Sulfur deficiency
- Zinc deficiency

---

## Project Structure
├── dataset/
│ ├── train/ # Training data (one folder per class)
│ ├── val/ # Validation data (one folder per class)
│
├── train_model.py # Script to train the MobileNetV2 classifier
├── predict_image.py # Script to run predictions on new images
├── best_mode.keras
├── final_model.keras
├── testcase0.jpg
├── README.md # Project documentation
├── LICENSE 

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

### Install dependencies
```bash
pip install tensorflow numpy matplotlib
```

## Usage

### Run Training:

```bash
python train_model.py   #train model
```
This will:
  -  Train MobileNetV2 on your dataset
  -  Save the best model as best_efficientnet_model.keras
  -  Save the final model as final_efficientnet_model.keras
  -  Plot training & validation accuracy/loss

### Run Predictions:

```bash
python predict_image.py #to classify a leaf image
```
Update the est image path in the scripts to where the test image maybe stored.

## Example Output:
  📷 Prediction for: leaf_sample.jpg
  Nitrogen deficiency: 0.8421
  Potassium deficiency: 0.1032
  Healthy: 0.0547


## License

This project is licensed under the Apache License 2.0.

You are free to: Use the code for personal, academic, or commercial purposes Modify, adapt, or improve the code Distribute the original or modified code

Conditions: You must include a copy of the Apache 2.0 license with any redistribution Retain copyright and attribution notices Clearly indicate if you modified any files The software is provided “as-is,” without any warranty

For the full license, see the LICENSE file.
