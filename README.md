
# Project: Tiny ImageNet CNN – From Scratch, Design to Cloud Deployment

## Introduction

This project demonstrates the end-to-end engineering of a convolutional neural network (CNN) trained from scratch on the Tiny ImageNet dataset, which consists of 200 classes with 64×64 images.

Tiny ImageNet contains 100,000 images across 200 classes, with each class having only 500 samples and image dimensions of 64×64. Due to the large number of classes, limited samples per class, small image size, and high intra-class variability, training a CNN from scratch on this dataset is particularly challenging.

For these reasons, the primary goal is **not** to achieve state-of-the-art performance, transfer learning can achieve higher accuracy. Instead, this project focuses on practical engineering: from data preprocessing, model design, and experimentation, to deployment on AWS.

The current trained model achieves a **validation accuracy of ~42%**, which is consistent with models trained from scratch on Tiny ImageNet. Further experimentation is ongoing, and validation accuracy will be updated as improvements are made.

---

## Folder Structure

```
requirements.txt           # Project dependencies
Training Logs/             # Experiment logs and metrics

src/
 ├── data_util/            # Data loaders, preprocessing, and augmentation
 ├── prototypes/           # Experimental model architectures
 └── deployment/
      ├── app.py           # Flask API for model inference
      ├── final_model.keras# Trained CNN model
      ├── class_names.json # Class label mapping file
      └── requirements.txt # Python dependencies for deployment
```

---

## Model Architecture


This architectural design was driven by the goal of creating a model that is lightweight, stable, and capable of extracting meaningful information from small inputs while remaining conservative with information. The following design decisions were made:

**Block 1:** Small filter size (2×2) was used to capture fine details from the input (64×64). While 3×3 filters are standard, they are better suited for larger inputs. Using smaller filters keeps the model lighter, which is crucial for a deep network. Batch normalization and LeakyReLU were applied to stabilize training. Average pooling was used instead of max pooling to maximize information preservation. Convolutional layers were padded to prevent shrinking of feature maps through multiple layers.

**Block 2:** Mild Spatial Dropout was introduced to help prevent overfitting. Bottleneck design (compress-expand) was used to efficiently extract features while keeping the network lightweight. No activation was applied immediately after the 1×1 compression layer to preserve the linearity of the compressed features. Applying a non-linearity at this stage could distort the linearly compressed information, reducing the effectiveness of the subsequent 3×3 convolution. After the 3×3 convolution, activations were applied during expansion to break linearity and improve feature representation.

**Block 3:** Follows the same bottleneck principle as Block 2. The receptive field increases progressively and reaches ~60×60 by the end of Block 3, covering almost the entire input.

**Block 4:** Global Average Pooling is used to reduce the number of parameters and summarize spatial information. Dropout (35%) is applied to further reduce overfitting risk.

---



---

## API Usage

The trained model is deployed as a REST API using Flask.

**Endpoint:**

```
POST http://3.28.60.161:5000/predict
```

Request Body:

* JSON format
* Image must be base64-encoded

```json
{
    "image": "<base64_encoded_image>"
}
```

**Response:**

```json
{
    "predicted_class": "<most_likely_class>",
    "top_predictions": [
        { "class": "<class_name_1>", "prob": <probability_1> },
        { "class": "<class_name_2>", "prob": <probability_2> },
        { "class": "<class_name_3>", "prob": <probability_3> }
    ]
}
```

**Python Example:**

```python
import requests
import base64
import json

# 1. API endpoint
url = "http://3.28.60.161:5000/predict"

# 2. Load and encode the image
with open("your_image_path.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode("utf-8")

# 3. Create the payload
payload = {"image": img_base64}

# 4. Send POST request
response = requests.post(url, json=payload)

# 5. Print the response
print(response.json())
```


---



## Setup & Installation

### **1. Training Setup**

These steps set up the environment to train the CNN from scratch using the compiled `model_training` module.

1. Clone the repository:

```bash
git clone https://github.com/Abdulaziz-Alrashidi/CNN_RNNAD_prototype
cd CNN_TinyImageNet
```

2. Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Update the dataset path in the training script at `src/model_training`:

```python
DATA_PATH = "/path/to/tiny-imagenet"  # replace with your dataset directory
train, val = get_dataset(DATA_PATH, image_size=(64,64), batch_size=64)
```

5. Run the training script:

```bash
python src/model_training.py
```

> **Note:** The training module is self-contained; it includes data loading, model definition, and training in one script.

---

### **2. Deployment Setup**

These steps set up the environment to run the Flask API for inference using the trained model.

1. Navigate to the deployment folder:

```bash
cd src/deployment
```

2. Create and activate a Python virtual environment (if not already activated):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install deployment-specific dependencies:

```bash
pip install -r requirements.txt
```

4. Start the Flask API:

```bash
nohup python3 app.py &
```

5. Use the API as described in the **API Usage** section.

---




---

## **Experimentation & Logs**

The logs contain basic comments on experiments. They do not include every minor tweak but instead serve as general guidelines for more significant changes.

---

## **Future Work**

* Experiment with improved architectures and regularization
* Update validation accuracy as experiments progress

---
