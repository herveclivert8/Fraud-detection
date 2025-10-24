# Financial Fraud Detection System 

## Overview

This project is practical method for building a financial fraud detection system. 
It focuses on a unified framework that uses recent advances in both fraud attack strategies and machine learning technique.
The system is designed in three main phases, each using a specific algorithm adapted to its task. 
Together, they help detect, group, and classify different types of fraud more effectively.

## Phase Model
### Anomaly detection (Autoencoder)
The first phase detects unusual or suspucious transactions using an Autoencoder.
This model helps to reduce false alerts by learning the normal patterns of financial activity.

### Clustering (K-means)
The second phase applies K-means clustering on the detected anomalies.
It groups transactions based on business-related signals, confirming whether they are truly fraudulent or not.

### Multi-class classification (LightGBM)
The third phase uses LightGBM to classify each confirmed fraud into a specific fraud category.
This ensures a clear identification of the type of fraud and allows better fraud tracking.

## Case study
The approach was tested in the banking sector, focusing on three major types of external fraud : 
- Money laundring
- Credit card fraud
- Mule account fraud

## Key benefis
- Fewer false alerts 
- Early detection of fraudulent behaviors
- Works well even with imbalanced data and missing labels
- Adapts to various fraud types and changing data

## Conclusion
This project shows how a hybrid machine learning approach can make fraud detection more accurate and reliable.
It provides a strong foundation for developing robust systems to protect banks and financial institutions from fraud. 
