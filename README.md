# MLOps Lab 3: Wine Classification API using FastAPI

This lab shows how to deploy a machine learning classification model as APIs using FastAPI. It uses the Wine Recognition dataset from scikit-learn and a Decision Tree classifier to predict wine categories based on 13 chemical measurements.

The lab extends the original FastAPI lab by implementing hyperparameter tuning, input validation, prediction probabilities, model metadata, optimized model loading, and automated API testing.

## Project Structure

```text
mlops_lab3/
└── fastapi_lab1/
    ├── assets/
    ├── model/
    │   └── wine_model.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── main.py
    │   ├── predict.py
    │   └── train.py
    ├── tests/
    │   ├── __init__.py
    │   └── test_api.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

The virtual environment (`fastapi_lab1_env/`) is created locally during setup and excluded from Git.

## Dataset

The project uses the Wine Recognition dataset available through scikit-learn.

| Property | Description |
|---|---|
| Dataset | Wine Recognition |
| Total samples | 178 |
| Input features | 13 numerical features |
| Target classes | 3 |
| Training samples | 142 |
| Testing samples | 36 |
| Task | Multiclass classification |

The dataset is split into 80% training and 20% testing data using stratified sampling to preserve class distributions.

## Prerequisites

The project was developed and tested on macOS using Python 3.13.18.

Required software:

- Python 3.13.18
- Git
- pip
- Terminal or command prompt
- VS Code or another code editor (optional)

### Python Dependencies

| Package | Version |
|---|---|
| FastAPI | 0.162.0 |
| Uvicorn | 0.54.0 |
| scikit-learn | 1.9.1 |
| Pandas | 3.1.0 |
| NumPy | 2.5.4 |
| Joblib | 1.6.0 |
| Pydantic | 2.14.1 |
| Pytest | 9.2.0 |
| HTTPX | 0.28.1 |

All required Python packages are specified in `requirements.txt`.

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/alekhyabulusu/mlops_lab3.git
cd mlops_lab3/fastapi_lab1
```

### Step 2: Create a Virtual Environment

For macOS/Linux:

```bash
python3.13 -m venv fastapi_lab1_env
source fastapi_lab1_env/bin/activate
```

For Windows:

```powershell
py -3.13 -m venv fastapi_lab1_env
fastapi_lab1_env\Scripts\activate
```

### Step 3: Install Dependencies

Install the required packages:

```bash
python -m pip install -r requirements.txt
```
## Running the Project

All commands should be executed from the `fastapi_lab1` directory with the virtual environment activated.

### Step 1: Load the Dataset

```bash
python -m src.data
```

Expected output:

```text
Wine Dataset Loaded Successfully!
Training samples: 142
Testing samples: 36
Number of features: 13
```

### Step 2: Train the Model

```bash
python -m src.train
```

The training script performs the following operations:

1. Loads the Wine dataset.
2. Splits the dataset into training and testing sets.
3. Performs hyperparameter tuning using GridSearchCV.
4. Uses five-fold stratified cross-validation.
5. Evaluates the best Decision Tree classifier.
6. Saves the trained model and metadata.

The model is saved to:

```text
model/wine_model.pkl
```

#### Training Results

| Metric | Result |
|---|---|
| Model | Decision Tree Classifier |
| Best criterion | Gini |
| Best max depth | 4 |
| Best min samples split | 2 |
| Cross-validation accuracy | 90.22% |
| Test accuracy | 94.44% |

The model correctly classified 34 out of 36 test samples.

### Step 3: Run a Standalone Prediction

```bash
python -m src.predict
```
Expected output:

```text
Prediction successful!
Predicted class: 0
Class name: class_0
Confidence: 1.0
Class probabilities: {'class_0': 1.0, 'class_1': 0.0, 'class_2': 0.0}
```

### Step 4: Start the FastAPI Server

```bash
python -m uvicorn src.main:app --reload
```

The API will be available at:

http://127.0.0.1:8000

Interactive Swagger documentation:

http://127.0.0.1:8000/docs

The `--reload` option is intended for local development.

## API Endpoints

The application provides four endpoints.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Displays API information |
| GET | `/health` | Checks model availability |
| GET | `/model-info` | Returns model metadata |
| POST | `/predict` | Predicts wine categories |

### 1. Root Endpoint

**GET /**

Returns a welcome message, API version, and available endpoints.

### 2. Health Check Endpoint

**GET /health**

Checks whether the trained model is loaded.

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Model Information Endpoint

**GET /model-info**

Returns model metadata, including:

- Model type
- Test accuracy
- Best hyperparameters
- Feature names
- Target class names

Example response:

```json
{
  "model_type": "DecisionTreeClassifier",
  "accuracy": 0.9444444444444444,
  "best_parameters": {
    "criterion": "gini",
    "max_depth": 4,
    "min_samples_split": 2
  },
  "feature_names": [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline"
  ],
  "target_names": [
    "class_0",
    "class_1",
    "class_2"
  ]
}
```

### 4. Prediction Endpoint

**POST /predict**

Accepts 13 numerical features describing the chemical properties of a wine sample.

Example request:

```json
{
  "features": {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280/od315_of_diluted_wines": 3.92,
    "proline": 1065
  }
}
```

Expected response:

```json
{
  "status": "success",
  "prediction": {
    "predicted_class": 0,
    "class_name": "class_0",
    "confidence": 1.0,
    "class_probabilities": {
      "class_0": 1.0,
      "class_1": 0.0,
      "class_2": 0.0
    }
  }
}
```

The confidence score represents the Decision Tree classifier's estimated probability for the predicted class.

### Input Validation

The API uses Pydantic to validate incoming requests.

Validation checks include:

- All 13 required features must be provided.
- Unexpected features are rejected.
- Feature values must be valid numerical inputs.

Requests containing missing or unexpected features return HTTP 422 validation errors.

## Automated Testing

Automated API tests are implemented using Pytest and FastAPI's TestClient.

Run:

```bash
python -m pytest tests/test_api.py -v
```

### Test Cases

| Test | Description | Expected Status |
|---|---|---|
| `test_root_endpoint` | Verifies root endpoint | 200 |
| `test_health_endpoint` | Verifies model availability | 200 |
| `test_model_info_endpoint` | Verifies model metadata | 200 |
| `test_valid_prediction` | Verifies successful prediction | 200 |
| `test_missing_feature` | Rejects missing features | 422 |
| `test_extra_feature` | Rejects unexpected features | 422 |
| `test_prediction_response_schema` | Verifies the `/predict` endpoint uses the `PredictResponse` schema in OpenAPI | 200 |


### Test Results

```text
tests/test_api.py::test_root_endpoint PASSED
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_model_info_endpoint PASSED
tests/test_api.py::test_valid_prediction PASSED
tests/test_api.py::test_missing_feature PASSED
tests/test_api.py::test_extra_feature PASSED
tests/test_api.py::test_prediction_response_schema PASSED

7 passed, 1 warning in 2.45s
```

All seven automated tests passed successfully.

## Modifications from the Original Lab

The original FastAPI tutorial demonstrates basic machine learning model deployment.

This project extends the implementation with the following modifications.

### 1. Dataset Replacement

Replaced the Iris dataset with the Wine Recognition dataset.

The new dataset contains 13 chemical features and three target classes.

### 2. Hyperparameter Tuning

Implemented GridSearchCV with five-fold stratified cross-validation.

The following Decision Tree parameters are tuned:

- `max_depth`
- `min_samples_split`
- `criterion`

This allows the training pipeline to select suitable model parameters rather than relying on default settings.

### 3. Improved Model Evaluation

Added accuracy evaluation and a classification report containing:

- Precision
- Recall
- F1-score
- Support

### 4. Model Metadata Storage

Instead of saving only the trained classifier, the model artifact contains:

- Trained model
- Feature names
- Target names
- Test accuracy
- Best hyperparameters

This metadata is reused during inference and exposed through the API.

### 5. Enhanced Prediction Functionality

Modified the prediction logic to return:

- Predicted class ID
- Class name
- Confidence score
- Probabilities for all three classes

The prediction function also preserves the feature order used during training.

### 6. Pydantic Input Validation

Implemented request validation using Pydantic.

The API checks for missing and unexpected features before generating predictions.

Invalid requests return HTTP 422 responses.

### 7. Additional FastAPI Endpoints

Added two endpoints:

**`/health`**

Checks whether the model is available.

**`/model-info`**

Returns model metadata, including accuracy, hyperparameters, and feature names.

### 8. Optimized Model Loading

Implemented FastAPI's lifespan functionality to load the trained model once during application startup.

The model is stored in `app.state` and reused for subsequent prediction requests.

This avoids repeatedly loading the serialized model from disk.

### 9. Automated API Testing

Added seven automated tests using Pytest and FastAPI's TestClient.

The tests verify endpoint responses, model availability, predictions, and validation errors.

These modifications extend the lab's core concept of deploying and serving machine learning models through FastAPI rather than only changing the dataset or model.

### 10. Pydantic Response Model

Implemented Pydantic response models (`Prediction` and `PredictResponse`) to define and validate the structure of API prediction responses.

The `/predict` endpoint now uses `response_model=PredictResponse`, extending the original tutorial's `IrisResponse` implementation.

The response model includes:
- Predicted class ID
- Class name
- Confidence score
- Class probabilities
- Prediction status

This modification provides automatic response validation, type consistency, and improved API documentation through Swagger UI.

## Results

The Decision Tree classifier achieved:

| Metric | Result |
|---|---|
| Cross-validation accuracy | 90.22% |
| Test accuracy | 94.44% |
| Correct test predictions | 34/36 |
| Automated tests passed | 7/7 |

The model was successfully trained, evaluated, serialized, loaded, and served through FastAPI.

All four API endpoints were manually tested, and all seven automated tests passed.
