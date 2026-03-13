# ML-Project-Fitness-Tracker

## **Description**    

This project aims to build a complete machine learning pipeline. The process begins with reading, cleaning, and processing raw data from CSV files, followed by visualizing the data as time series to understand its underlying patterns. Advanced outlier detection techniques, including Chauvenet’s Criterion and Local Outlier Factor (LOF), are then applied to improve data quality. After that, a comprehensive feature engineering phase is performed using techniques such as low-pass filters, Principal Component Analysis (PCA), and clustering. For predictive modeling, several machine learning algorithms—including Naive Bayes, Support Vector Machines (SVM), Random Forests, and Neural Networks—are trained and compared to achieve high prediction accuracy. Finally, the project concludes with the development of a custom algorithm designed to automatically and accurately count movement repetitions.

---

## **Dataset : MetaMotion Physical Activity Analysis**
1. **Overview**
The dataset comprises **187 raw CSV files** generated from **MetaMotion wearable sensors**. It captures high-resolution movement data across various resistance training exercises, designed for activity recognition and repetition counting tasks.
2. **Sensor Specifications**
The data is collected using two primary inertial measurement units (IMUs):
    * 3-Axis Accelerometer: Captures linear acceleration in $g$ units ($x, y, z$ axes) at a frequency of 12.5Hz.
    * 3-Axis Gyroscope: Captures angular velocity in $deg/s$ ($x, y, z$ axes) at a frequency of 25Hz.
3. **Metadata & Naming Convention**
Each file follows a strict naming convention that acts as a primary source for automated data labeling. The metadata is encoded as follows:
[Participant]-[Label]-[Category][SetNumber]-[SensorType].csv
- **Participants**: 5 distinct individuals (Identified as A, B, C, D, E).
- **Activity Labels**:
    - bench: Bench Press
    - squat: Squat
    - ohp: Overhead Press
    - dead: Deadlift
    - row: Barbell Row
    - rest: Stationary/Non-active period.
- **Categories (Intensity)**:
    - Heavy Set: High-intensity sessions containing 5 repetitions.
    - Medium Set: Moderate-intensity sessions containing 10 repetitions.
- **Set Number**: Tracks the sequence of sets for the same exercise.
4. **Data Features**
Each record in the dataset contains the following features:
- epoch (ms) | Unix timestamp in milliseconds
- time | Formatted date and time | YYYY-MM-DDTHH:MM:SS ||
- x-axis | Acceleration or Angular Velocity on X-axis | $g$ or $deg/s$ ||
- y-axis | Acceleration or Angular Velocity on Y-axis | $g$ or $deg/s$ ||
- z-axis | Acceleration or Angular Velocity on Z-axis | $g$ or $deg/s$ |
5. **Data Challenges & Objectives**
- Multi-Frequency Handling: Synchronizing the 12.5Hz Accelerometer data with the 25Hz Gyroscope data.
- Outlier Removal: Cleaning noise using statistical methods like Chauvenet’s Criterion.
- Feature Engineering: Extracting meaningful patterns from raw temporal data using PCA and Frequency Domain analysis.
