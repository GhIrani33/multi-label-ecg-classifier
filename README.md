## **ECG Multi-Label Classification with Deep Learning**

### **Project Description**
This project focuses on multi-label classification of ECG signals using deep learning techniques, including Conv1D layers with residual connections, Bidirectional LSTMs, and attention mechanisms. The task involves predicting multiple heart-related conditions from ECG signals, aiming to develop an automatic ECG interpretation system that can assist cardiologists in diagnosing heart diseases.

### **Dataset**
We use the **[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)** dataset, a large publicly available electrocardiography dataset published by Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Wojciech Samek, and Tobias Schaeffter. The dataset consists of 21,799 clinical 12-lead ECGs from 18,869 patients, with each ECG lasting 10 seconds. The dataset includes multiple diagnostic labels per record, following the SCP-ECG standard.

#### **Key Details**:
- **Version**: 1.0.3
- **Sampling Frequency**: 500 Hz (downsampled version available at 100 Hz).
- **Leads**: Standard 12 leads (I, II, III, AVR, AVL, AVF, V1–V6).
- **Labels**: 5 diagnostic categories - NORM, MI, STTC, CD, HYP.
- **Splits**: Recommended train-test splits from the dataset were used (folds 1-8 for training, fold 9 for validation, fold 10 for testing).

#### **Citations**:
Please cite the following if using the dataset:
- **Dataset Citation**: Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). PTB-XL, a large publicly available electrocardiography dataset (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45.
- **Original Publication**: Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., Schaeffter, T. (2020), PTB-XL: A Large Publicly Available ECG Dataset. Scientific Data. https://doi.org/10.1038/s41597-020-0495-6.

### **Model Overview**

These architectures are widely used in ECG analysis and are supported by various research publications. The combination of these techniques allows for effective multi-label classification of ECG signals. For more details on the model, refer to the **models.py** file.

#### **Key Components**:

- **Convolutional Layers with Residual Connections**: 
  - The model uses **Conv1D layers** to capture spatial features from ECG signals. Residual connections help in stabilizing the training of deep networks by providing shortcut paths for gradients.
  
- **Bidirectional LSTM**:
  - A **Bidirectional LSTM** is used to capture temporal dependencies in the ECG signals. It helps the model learn information from both past and future time steps.

- **Attention Mechanism**:
  - A custom **attention layer** is applied after the LSTM layer to help the model focus on the most relevant parts of the ECG signal. This mechanism learns the importance of each timestep and adjusts the model’s focus accordingly.

- **Metadata Input**:
  - Apart from the ECG signal, metadata (such as age and sex) is processed through a separate **Dense network** and concatenated with the ECG features for classification.

- **Focal Loss with Class Weights**:
  - The model is trained using a custom **Focal Loss function** that adjusts for class imbalance by giving more weight to harder-to-classify samples. This helps improve performance on underrepresented classes.



### **Hyperparameters**
The model is fine-tuned using hyperparameters found via optimization. 

### **Training**
The model is trained using **Adam optimizer** with a learning rate of 4.67e-4 and the following callbacks:
- **Early Stopping**: Monitors the validation AUC and stops training when improvement plateaus.
- **Model Checkpoint**: Saves the best model based on validation AUC.
- **ReduceLROnPlateau**: Reduces the learning rate when validation performance plateaus to fine-tune learning.


### **Performance on Test Set**
- **Precision (macro):** 0.74
- **Recall (macro):** 0.74
- **F1-score (macro):** 0.74
- **AUC-ROC (macro):** 0.91
- **Subset Accuracy:** 61.56%
- **Hamming Loss:** 11.58%

### **Confusion Matrices**
Confusion matrices are generated for each diagnostic class (e.g., NORM, MI, STTC, CD, HYP) to assess the performance in detail.

### **How to Use**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/ecg-multilabel-classification.git
   ```

2. **Install Dependencies**:
   - Create a virtual environment (optional but recommended):
     ```bash
     python3 -m venv venv
     source venv/bin/activate   # For Windows: venv\Scripts\activate
     ```
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Training Script**:
   ```bash
   python training.py
   ```

4. **Model Inference**:
   - You can use the saved model (`Final-ECG-Model.h5`) for inference on new ECG data. Just load the model using `keras.models.load_model()` and run predictions.
  

   ### Running the Tests

   This project includes unit tests for preprocessing and model functions to ensure code reliability. You can run the tests locally or as part of Continuous Integration (CI) via GitHub Actions.

   **Locally:**
   To run all tests locally, use the following command:

   ```bash
   python -m unittest discover -s tests
   ```

   This will execute all test cases in the `tests/` folder.

   **Continuous Integration:**
   The tests are also automatically executed on every push or pull request using GitHub Actions, ensuring that the project stays in good working condition.



You will run the tests using the following steps:

1. **Ensure you have your testing framework set up:**
   Since you’re using `unittest` in Python, the `unittest` module should already be included in your Python installation. Ensure that your project is structured correctly with the `tests/` folder containing `test_preprocessing.py` and `test_models.py`.

2. **Run the tests locally:**
   You can run all the tests in your `tests/` folder by executing this command in your terminal:

   ```bash
   python -m unittest discover -s tests
   ```

   This command will automatically find and run all test files in the `tests/` folder.

3. **Run tests automatically using GitHub Actions (CI):**
   Since you’ve already set up GitHub Actions in your `.github/workflows/python-app.yml` file, the tests will run automatically every time you push new changes to your repository.

4. **Updating the README File:**
   It is a good idea to include a section in the README explaining how to run tests. You can add a section like this:

## Running Tests

This repository includes automated tests to ensure the model and preprocessing pipelines work correctly.

## How to Run Tests Locally

To run the tests locally, first make sure you have all the necessary dependencies installed, then simply run:

```bash
python -m unittest discover -s tests


**Why Use Original ECG Signals?**
Through extensive testing, including various preprocessing techniques and signal visualization, we found that using the **original ECG signals** without heavy modification produced the most robust results. While alternative preprocessing approaches gave higher performance during testing, they were less reliable in real-world applications. Thus, for this project, we prioritize **real-world robustness** over purely performance-driven metrics.

**Contributing**
We welcome contributions to this project! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure that all tests pass.
4. Submit a pull request.

**License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
