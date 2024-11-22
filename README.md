# DEPI_DoctorAssistant_GraduationProject
## **Brain Tumor Detection and Ocular Disease Classification**

![Project Banner](https://i.imgur.com/vWY5kOR.png)

## **Table of Contents** 📋
1. [📖 Introduction](#introduction)
2. [✨ Features](#features)
3. [💻 Technologies Used](#technologies-used)
4. [📊 Dataset](#dataset)
5. [🔍 Model](#model)
6. [⚙️ Installation](#installation)
7. [🚀 Usage](#usage)
8. [🔮 Future Work](#future-work)
9. [🤝 Contributing](#contributing)
10. [📜 License](#license)


---

## **Introduction**
This project leverages **machine learning** and **deep learning** techniques to detect and classify medical conditions from **MRI brain scans** and **ocular images**, providing an efficient and accurate tool to assist healthcare professionals in diagnosing complex diseases. Early and precise detection of such conditions is crucial for timely intervention, better treatment outcomes, and improved patient care.

#### **Motivation**
Brain tumors and ocular diseases are among the most critical health challenges, requiring specialized expertise for diagnosis. Limited access to trained professionals in certain regions, combined with the potential for human error, highlights the need for automated diagnostic solutions. This project bridges the gap by offering a reliable, AI-driven system that can analyze medical images and provide predictions, empowering clinicians with decision-support tools.

#### **Key Features**
1. **Brain Tumor Classification**:
   - Categorizes brain tumors into **Gliomas**, **Meningioma**, **Pituitary**, and **No Tumor** using MRI scans.
   - Employs advanced transfer learning techniques for high accuracy.
   - Provides visual explanations for predictions, enhancing transparency and trust.

2. **Ocular Disease Detection**:
   - Identifies various eye diseases, including **Myopia**, **Macular Degeneration**, **Glaucoma**, **Diabetic Retinopathy**, **Hypertensive** and **Cataracts**.
   - Offers potential for extension to more diseases and improved functionalities.

3. **Scalable and Modular**:
   - Built with scalability in mind, allowing for the integration of additional diseases, modalities, and data sources.
   - Supports continuous learning through retraining on new datasets.

#### **Significance**
The project stands at the intersection of technology and healthcare, demonstrating how AI can enhance diagnostic accuracy while reducing the workload on healthcare providers. By automating routine and complex analysis tasks, this solution can:
- Reduce diagnostic errors.
- Provide accessibility in under-resourced medical facilities.
- Enable faster treatment decisions through real-time analysis.

#### **Future Directions**
As the project evolves, we aim to:
- Extend disease detection capabilities to include more neurological and ophthalmological conditions.
- Implement advanced techniques such as **segmentation** to highlight affected regions in medical images.
- Develop multilingual support and better accessibility for a global audience.
- Integrate with wearable devices and IoT platforms for real-time monitoring and analysis.

In essence, this project represents a step towards **AI-assisted healthcare**, fostering a future where cutting-edge technology aids in saving lives and improving quality of care.

---

## **Features**
![Example of Uploading an Image](https://i.imgur.com/KworzUN.jpeg)
![Example of Uploading an Image](https://i.imgur.com/lUPP3mc.jpeg)
![Example of Uploading an Image](https://i.imgur.com/6ahze33.jpeg)
  
- AI-powered classification models with high accuracy.
- A dataset of 7000 image of brain tumor MRI.
- Streamlit-based user-friendly interface for end-users.
- Integration with Azure Q&A functionality for patient education.
- Accuracy up to 99%

![Example of Uploading an Image](https://i.imgur.com/wT9zlmq.jpeg)
![Example of Uploading an Image](https://i.imgur.com/6Tapwup.jpeg)
![Example of Uploading an Image](https://i.imgur.com/mC8pOZu.jpeg)

- AI-powered classification models with high accuracy.
- A dataset of around 6500 image of Ocular Disease MRI augmented to around 80,000.
- Flask-based user-friendly interface for end-users.
- Integration with Azure Q&A functionality for patient education.
- Accuracy up to 99%
---

## **Technologies Used**
- **Programming Language**: Python  
- **Libraries**:  
  - TensorFlow/Keras  
  - Sklearn
  - Streamlit  
  - NumPy & Pandas  
  - Matplotlib & Seaborn
  - Flask
  - Azure
  - SQLite
- **Model Architecture**: Xception, VGG19
- **Deployment**: Streamlit, Flask, Azure for chatbot/Q&A integration

---

## **Dataset**
### Brain Tumor Detection:
- Source: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
- Preprocessing: Images are resized, normalized, and augmented to improve model performance.

### Ocular Disease Classification:
- Source: [Kaggle ODIR](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k/data)  and Other Real life resources.
- Preprocessing: Images are resized, normalized, and augmented to improve model performance.
---

## **Model**
### Brain Tumor Classification:
- **Model Architecture**: Transfer learning using the Xception model.
- **Classes**: Gliomas, Meningioma, Pituitary, No tumor.  

### Ocular Disease Classification:
- **Model Architecture**: VGG19 for specific eye diseases.
- **Classes**: Myopia, Glaucoma, Cataracts, and more.

#### **Training and Evaluation:**
- Loss Function: Categorical Crossentropy  
- Optimizer: Adam, Adamax  
- Metrics: Accuracy, Precision, Recall, F1 Score  

---

## **Installation**
### Prerequisites:
- Python 3.8+
- Anaconda/Miniconda (Optional)

### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/7ekmaz/DEPI_DoctorAssistant_GraduationProject.git
    ```
2. Navigate to the project directory:
    ```bash
    cd DEPI_DoctorAssistant_GraduationProject
    ```

---

## **Usage**
### Running the Application:
1. Launch the Streamlit/Flask app:
    ```bash
    streamlit run app.py
    ```
2. Upload an MRI or eye image for prediction.
3. View predictions and additional resources through the chatbot/Q&A feature.


---

## **Future Work**

This project has the potential to expand and improve in various areas. Some ideas for future work include:

1. **Expanding Disease Coverage**:
   - Adding support for more brain and ocular diseases.
   - Extending to other medical domains such as cardiovascular and pulmonary diseases.

2. **Improved Explainability**:
   - Implementing Grad-CAM or similar visualization techniques to highlight affected regions in medical images.

3. **Segmentation Capabilities**:
   - Developing tumor and disease segmentation to localize the affected areas more precisely.

4. **Real-Time Deployment**:
   - Deploying the model on edge devices or mobile applications for real-time diagnosis.

5. **Integration with Wearables**:
   - Enabling data collection and monitoring through wearable devices, integrating IoT for healthcare.

6. **Multilingual Support**:
   - Adding support for multiple languages to improve accessibility for a global audience.

7. **Continuous Learning**:
   - Implementing online learning to allow the model to improve over time with new data.

We welcome contributions to help bring these features to life. 


---

## **Contributing**

We welcome and appreciate contributions to this project! To get started, follow these steps:

---

## **License**

This project is licensed under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute this project under the terms of the license.

---



