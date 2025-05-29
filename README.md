# Human Activity Recognition using Wearable Sensor Data

![Activity Recognition Banner](images/activity_banner.png)

## Project Overview

This project implements and evaluates machine learning models for human activity recognition using wearable sensor data. The goal is to accurately classify different physical activities (walking, sitting, standing, etc.) based on accelerometer and gyroscope measurements.

## Dataset

The dataset contains recordings from wearable sensors attached to 30 subjects performing six different activities:
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

Each record includes 561 features derived from the raw sensor data, including time and frequency domain variables.

## Key Visualizations

### Activity Distribution

The dataset contains a balanced distribution of activities across subjects:

![Activity Distribution](images/activity_distribution.png)

### Train-Validation Split

We used an 80:20 stratified split to ensure representative distribution of activities in both training and validation sets:

![Train-Validation Split](images/train_val_distribution.png)

### Dimensionality Reduction

#### PCA Visualization

Principal Component Analysis (PCA) was used to reduce the dimensionality of the feature space while preserving 95% of the variance:

![PCA Components](images/pca_variance.png)

The first two principal components show some separation between activities:

![PCA Scatter Plot](images/pca_scatter.png)

#### LDA Visualization

Linear Discriminant Analysis (LDA) provides better separation between activity classes:

![LDA Scatter Plot](images/lda_scatter.png)

### Model Performance

We evaluated three SVM-based approaches:

![Model Comparison](images/model_comparison.png)

The per-activity performance varies across models:

![Per-Activity Performance](images/activity_performance_radar.png)

### Confusion Matrices

#### SVM on Raw Features

![Raw SVM Confusion Matrix](images/raw_svm_confusion.png)

#### SVM with PCA

![PCA SVM Confusion Matrix](images/pca_svm_confusion.png)

#### SVM with LDA

![LDA SVM Confusion Matrix](images/lda_svm_confusion.png)

## Key Findings

1. **Overall Performance**: 
   - SVM with LDA achieved the highest accuracy at 96.2%
   - Dimensionality reduction significantly improved computational efficiency while maintaining high accuracy

2. **Activity-Specific Performance**:
   - Static activities (sitting, standing, laying) were generally easier to classify
   - Dynamic activities (walking, walking upstairs, walking downstairs) showed some confusion between them

3. **Hyperparameter Selection**:
   - The optimal parameters varied across models, with LDA-based models generally requiring lower C values

## Implementation

The project is implemented in Python using the following libraries:
- scikit-learn for machine learning algorithms
- pandas for data manipulation
- matplotlib and seaborn for visualization
- numpy for numerical operations

## Repository Structure

```
├── activity_recognition_notebook.ipynb  # Main notebook with all analysis
├── data/
│   ├── wearables_signal.csv            # Raw signal data
│   ├── wearables_activity.csv          # Activity labels
│   └── wearables_subject.csv           # Subject identifiers
├── images/                             # Visualizations from the notebook
└── README.md                           # This file
```

## Getting Started

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook activity_recognition_notebook.ipynb
```

## Future Work

1. **Deep Learning Approaches**: Explore CNNs or RNNs for automatic feature learning
2. **Feature Selection**: Investigate which features contribute most to classification accuracy
3. **Ensemble Methods**: Combine multiple models for improved performance
4. **Real-time Implementation**: Evaluate models in a streaming data context

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset is based on the Human Activity Recognition Using Smartphones Dataset
- Special thanks to all contributors and the open-source community
