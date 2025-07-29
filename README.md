# **Plant Disease Classification with ResNet50 (TensorFlow/Keras)**

---

### **1. Introduction**
This project implements a deep learning solution for classifying plant diseases from images using a fine-tuned ResNet50 convolutional neural network. The aim is to accurately identify various plant health conditions to assist in early detection and management of diseases. The project includes scripts for data loading and augmentation, model training, evaluation, and prediction on new images, demonstrating a complete machine learning pipeline.

---

### **2. Features**
* **Deep Learning Model:** Utilizes a pre-trained ResNet50 architecture, fine-tuned on a custom dataset for plant disease classification.
* **TensorFlow/Keras Framework:** Built and trained using the powerful and flexible TensorFlow and Keras libraries.
* **Data Preprocessing & Augmentation:** Employs `ImageDataGenerator` for efficient data loading and various augmentation techniques (rotation, shifts, zoom, brightness, flips) to enhance model generalization.
* **Class Weighting:** Incorporates class weights during training to handle imbalanced datasets, improving performance on minority classes.
* **Advanced Callbacks:** Integrates `EarlyStopping` (for preventing overfitting), `ReduceLROnPlateau` (for adaptive learning rate), and `ModelCheckpoint` (for saving the best model).
* **TensorBoard Integration:** Logs training metrics for visualization and analysis using TensorBoard.
* **Model Evaluation:** Provides a script to evaluate the trained model's performance on a validation set.
* **Flexible Prediction:** Scripts for making predictions on single images or batches of images from a folder.

---

### **3. Technologies Used**
This project leverages the following key technologies:

* **Programming Language:**
    * Python 3.x
* **Core Libraries/Frameworks:**
    * **TensorFlow:** The primary deep learning framework.
    * **Keras:** High-level API for building and training neural networks (integrated with TensorFlow 2.x).
    * **NumPy:** For numerical operations and data handling.
    * **Scikit-learn:** Used for `compute_class_weight` to handle class imbalance.
    * **TensorBoard:** For visualizing training progress and model graphs.

---

### **4. Getting Started**
Follow these instructions to set up and run the project on your local machine.

#### **4.1. Prerequisites**
Ensure you have the following software installed:

* **Python 3.x:** Download from [python.org](https://www.python.org/). It's recommended to use Python 3.8 or newer.
* **Git:** Download from [git-scm.com](https://git-scm.com/downloads).

#### **4.2. Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```
    (Replace `[your-username]` and `[your-repo-name]` with your actual GitHub details, e.g., `plant-disease-classification`)

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * **For GPU acceleration (if you have an NVIDIA GPU):**
        Ensure you have compatible CUDA Toolkit and cuDNN installed. Then, install the GPU version of TensorFlow: `pip install tensorflow[and-cuda]`. Check TensorFlow's official installation guide for specific instructions based on your system.

#### **4.3. Download the Dataset**
The image dataset used for training and testing is too large to host on GitHub. You will need to download it separately.

1.  **Dataset Source:** [Provide a link to your dataset here, e.g., Kaggle, Google Drive, or another data hosting platform].
    * **Example (Hypothetical):** [https://www.kaggle.com/datasets/example/plant-disease-dataset](https://www.kaggle.com/datasets/example/plant-disease-dataset)
2.  **Extract:** After downloading, extract the dataset.
3.  **Place Data:** Create a `data/` directory in the root of this project and place the `train/` and `test/` (or `validation/`) subdirectories containing your images inside it.
    ```
    [your-repo-name]/
    ├── data/
    │   ├── train/
    │   │   ├── class1_folder/
    │   │   └── class2_folder/
    │   └── test/
    │       ├── class1_folder/
    │       └── class2_folder/
    ```

---

### **5. Usage**

This project provides scripts for training, evaluating, and predicting plant diseases.

#### **5.1. Training the Model**

1.  **Activate your virtual environment.**
2.  **Navigate to the root of the project directory.**
3.  **Run the training script:**
    ```bash
    python train.py
    ```
    * **Optional arguments:**
        * `--batch_size <int>`: Specify the training batch size (default: 32).
        * `--epochs <int>`: Specify the number of training epochs (default: 50).
        * `--learning_rate <float>`: Set the initial learning rate (default: 0.0005).
    * **Example with arguments:**
        ```bash
        python train.py --batch_size 16 --epochs 100 --learning_rate 0.001
        ```
    * Training progress will be displayed in your terminal. The best model will be saved in a `models/` directory (created automatically). TensorBoard logs will be saved in `logs/`.

#### **5.2. Launching TensorBoard**

To monitor training progress, loss, and accuracy curves:

1.  **Activate your virtual environment.**
2.  **Navigate to the root of the project directory.**
3.  **Run TensorBoard:**
    ```bash
    tensorboard --logdir logs
    ```
    TensorBoard will provide a local URL (e.g., `http://localhost:6006/`) to open in your web browser.

#### **5.3. Evaluating the Model**

To evaluate the performance of your trained model on the test/validation set:

1.  **Activate your virtual environment.**
2.  **Navigate to the root of the project directory.**
3.  **Run the evaluation script:**
    ```bash
    python evaluate.py
    ```
    *(Note: Ensure `evaluate.py` points to the correct model file (`best_plant_disease_model.keras`) and validation directory (`data/test` or `data/validation`) for consistency.)*

#### **5.4. Making Predictions**

To predict plant diseases on new images:

1.  **Activate your virtual environment.**
2.  **Navigate to the root of the project directory.**
3.  **For single image prediction (using `test.jpg`):**
    * Place `test.jpg` in the root directory or update the `test_image` path in `test.py`.
    ```bash
    python test.py
    ```
4.  **For batch image prediction (using `test_images/` folder):**
    * Place your test images inside the `test_images/` directory.
    ```bash
    python test.py
    ```

---

### **6. Screenshots/Demos**
*(Replace these with actual links to your images or GIFs)*

* **Training Process (TensorBoard):** A screenshot of your TensorBoard dashboard showing accuracy and loss curves.
    ![TensorBoard Training Metrics](https://via.placeholder.com/700x400?text=TensorBoard+Metrics)

* **Sample Prediction:** A screenshot showing a sample image with its predicted class and confidence (can be a custom visualization if you create one).
    ![Sample Prediction](https://via.placeholder.com/700x400?text=Plant+Disease+Prediction+Example)

* **Model Architecture (Optional):** A screenshot of the model summary or graph from TensorBoard if it's visually informative.
    ![Model Architecture](https://via.placeholder.com/700x400?text=ResNet50+Model+Summary)

---

### **7. Project Structure**

### **8. Contributing**
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

### **9. License**
This project is licensed under the **MIT License** - see the `LICENSE` file for details. (Create a `LICENSE` file in the root of your project if you don't have one).

---

### **10. Contact**
[Your Name] - [your.email@example.com]
Project Link: [https://github.com/your-username/your-repo-name](https://github.com/your-username/your-repo-name)

---
