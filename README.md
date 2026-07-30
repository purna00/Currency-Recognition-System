\# Currency Recognition System Using Image Processing



A deep learning-based system for recognizing Indian currency denominations from images.



\## Supported Currency Classes



\- ₹10

\- ₹20

\- ₹50

\- ₹100

\- ₹200

\- ₹500



\## Technologies Used



\- Python

\- TensorFlow

\- Keras

\- MobileNetV2

\- OpenCV

\- NumPy

\- Scikit-learn

\- Streamlit

\- Matplotlib



\## Methodology



1\. Dataset collection

2\. Image preprocessing

3\. Image resizing

4\. Data augmentation

5\. MobileNetV2 transfer learning

6\. Model fine-tuning

7\. Currency classification

8\. Model evaluation

9\. Streamlit deployment



\## Model



MobileNetV2 with transfer learning and fine-tuning was used for currency classification.



\## Results



Validation Accuracy: 85.19%



Validation dataset size: 270 images



Macro F1-score: 84.98%



The model achieved its best class-wise F1-score for ₹500 at 92.78%.



\## Application



The project includes a Streamlit web application that allows users to upload a currency image and receive:



\- Predicted denomination

\- Prediction confidence

\- Class-wise prediction probabilities



\## How to Run



\### Install dependencies



```bash

pip install -r requirements.txt

