# Currency Recognition System Using Image Processing

A deep learning-based system for recognizing Indian currency denominations from images.

## Supported Currency Classes

* ₹10
* ₹20
* ₹50
* ₹100
* ₹200
* ₹500

## Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* OpenCV
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib

## Methodology

1. Dataset collection
2. Image preprocessing
3. Image resizing
4. Data augmentation
5. MobileNetV2 transfer learning
6. Model fine-tuning
7. Currency classification
8. Model evaluation
9. Application interface development

## Model

MobileNetV2 with transfer learning and fine-tuning was used for currency classification.

## Results

* Validation Accuracy: **85.19%**
* Validation Dataset Size: **270 images**
* Macro F1-score: **84.98%**
* Best Class-wise F1-score: **92.78% for ₹500**

## Application

The project includes an application interface that allows users to process a currency image and obtain:

- Predicted denomination
- Prediction confidence
- Class-wise prediction probabilities

## Sample Prediction

The model predicts the currency denomination along with its confidence score.

## Model Evaluation

The evaluation script generates:

- Validation accuracy
- Classification report
- Confusion matrix
- Macro F1-score

## Project Files

- `train.py` - Trains the currency recognition model
- `predict.py` - Predicts the denomination of a currency image
- `evaluate_model.py` - Evaluates the trained model and generates performance metrics
- `app.py` - Application interface
- `model/` - Contains the trained model
- `requirements.txt` - Contains the required Python dependencies
## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run prediction

```bash
python predict.py
```

Enter the path of a currency image when prompted.

### Evaluate the model

```bash
python evaluate_model.py
```

The evaluation script generates:

* Validation accuracy
* Classification report
* Confusion matrix
* Macro F1-score

## Conclusion

The Currency Recognition System successfully classifies six Indian currency denominations using deep learning and computer vision techniques. The MobileNetV2-based model achieved a validation accuracy of **85.19%** and a macro F1-score of **84.98%** on a validation dataset of 270 images.

## Future Scope

The system can be further improved by increasing the dataset size, adding more currency denominations, improving image quality and preprocessing, and deploying the application as a web or mobile-based solution.
