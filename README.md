# Char_Recognizer Task

## Overview
This project focuses on solving the **Street View Character Recognition Task**, which involves recognizing house numbers from street view images. The task is framed as a character recognition problem in computer vision, leveraging a dataset derived from real-world scenarios.

## Dataset
The dataset is sourced from the **Street View House Numbers Dataset (SVHN)** and has been processed for the competition. It includes:
- **Training Set**: 30,000 images with RGB data, encoded labels, and bounding box information.
- **Validation Set**: 10,000 images with the same format as the training set.
- **Test Set**: 40,000 images, provided without label information.

**Example Data Format**:
```json
"000001.png": {
  "height": [32, 32],
  "label": [2, 3],
  "left": [77, 98],
  "top": [29, 25],
  "width": [23, 26]
}
```

## Baseline Method
A baseline implementation baseline.ipynb is provided to guide participants, which includes:
1. **Dataset Preparation**:
   - Download and decompress the dataset.
   - Process the training and validation sets.
   - Handle empty labels by encoding them as class `10`.
   
2. **Model Design**:
   - The task is converted into a multi-digit classification problem, where each digit is classified as `0-9` or "empty".
   - Only the first four digit positions are considered for prediction.
   
3. **Loss Function**:
   - A **Cross-Entropy Loss** is used with label smoothing. For example:
     - Original label: `[1, 0, 0]`.
     - Smoothed label: `[0.9, 0.05, 0.05]`.
   - Benefits:
     1. Improves model generalization.
     2. Reduces overfitting and overconfidence.
     3. Increases robustness to label noise.
   
4. **Training and Validation**:
   - Training involves basic steps of gradient zeroing, loss computation, backpropagation, and parameter updates.
   - Validation is used to evaluate performance on unseen data.

5. **Evaluation**:
   - Submit predictions on the test set for evaluation.

## Improvement Ideas
Participants are encouraged to improve upon the baseline referencing the following strategies:
1. **Model Enhancements**:
   - Use deeper or more advanced convolutional neural networks (CNNs) or other architectures.
   - Explore object detection models (e.g., YOLO series) to treat digits as separate detection classes.
   - Leverage pre-trained scene text detection/recognition models and fine-tune them on the dataset, or train them from the random initialization.

2. **Data Augmentation**:
   - Experiment with data augmentation techniques to improve generalization.

3. **Loss Function and Hyperparameter Tuning**:
   - Adjust the loss function and optimize hyperparameters to improve performance.

4. **Model Ensemble**:
   - Combine predictions from multiple models using weighted voting or other ensemble techniques.


## Submission Requirements
Participants must submit the following:
1. **Test Set Results**:
   - Submit results on the test set through the official platform for evaluation. 
   - Scoring:
     - ≥ 0.86: +1 point
     - ≥ 0.88: +2 points
     - ≥ 0.90: +3 points
     - ≥ 0.92: +4 points
2. **Implementation Report**:
   - Provide a detailed report (up to 4 pages) covering:
     - Environment setup.
     - Model design and loss function explanation.
     - Strategies for improving test set performance (e.g., data augmentation, parameter tuning).
     - Innovations or modifications made to existing methods.
     - Challenges encountered and how they were addressed.
3. **Code**:
   - Submit the complete implementation code (excluding the dataset).

**Submission Format**:
- Name the submission file as `StudentID_Name_PJ1.zip` 
- Submit to the elearning platform by **April 21, 2025, 23:59**.

## References
- [Tianchi Competition Platform](https://tianchi.aliyun.com/competition/entrance/531795)
- [YOLO Series GitHub Repository](https://github.com/ultralytics/ultralytics)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/index.html)
- [OpenOCR GitHub Repository](https://github.com/Topdu/OpenOCR?tab=readme-ov)

## Scoring
The final score is calculated as the sum of:

- Test Set Results: Maximum 4 points.
- Implementation Introduction: Maximum 2 points.
- Improvements and Innovations: Maximum 3 points.
  
If the total score exceeds 8 points, it will be capped at 8 points.

Good luck with your project!
