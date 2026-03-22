# ML-Project-Fitness-Tracker

## **Description**    

This project aims to build a complete machine learning pipeline. The process begins with reading, cleaning, and processing raw data from CSV files, followed by visualizing the data as time series to understand its underlying patterns. Advanced outlier detection techniques, including Chauvenet’s Criterion and Local Outlier Factor (LOF), are then applied to improve data quality. After that, a comprehensive feature engineering phase is performed using techniques such as low-pass filters, Principal Component Analysis (PCA), and clustering. For predictive modeling, several machine learning algorithms—including Naive Bayes, Support Vector Machines (SVM), Random Forests, and Neural Networks—are trained and compared to achieve high prediction accuracy. Finally, the project concludes with the development of a custom algorithm designed to automatically and accurately count movement repetitions.

---

## **Part 1: Dataset MetaMotion Physical Activity Analysis**
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
       * bench: Bench Press
       * squat: Squat
       * ohp: Overhead Press
       * dead: Deadlift
       * row: Barbell Row
       * rest: Stationary/Non-active period.
   
   - **Categories (Intensity)**:
       * Heavy Set: High-intensity sessions containing 5 repetitions.
       * Medium Set: Moderate-intensity sessions containing 10 repetitions.
     
   - **Set Number**: Tracks the sequence of sets for the same exercise.
4. **Data Features**
Each record in the dataset contains the following features:
   - epoch (ms) | Unix timestamp in milliseconds
   - time | Formatted date and time | YYYY-MM-DDTHH:MM:SS |
   - x-axis | Acceleration or Angular Velocity on X-axis | $g$ or $deg/s$ 
   - y-axis | Acceleration or Angular Velocity on Y-axis | $g$ or $deg/s$ 
   - z-axis | Acceleration or Angular Velocity on Z-axis | $g$ or $deg/s$
5. **Data Challenges & Objectives**
   - Multi-Frequency Handling: Synchronizing the 12.5Hz Accelerometer data with the 25Hz Gyroscope data.
   - Outlier Removal: Cleaning noise using statistical methods like Chauvenet’s Criterion.
   - Feature Engineering: Extracting meaningful patterns from raw temporal data using PCA and Frequency Domain analysis.

---

## **Part 2: Data Processing & Integration**

1. **Data Aggregation & Metadata Extraction**
The first step involved parsing 187 individual CSV files. A custom script was developed to iterate through the data directory and extract key features from filenames, including:
   - Participant ID.
   - Exercise Label.
   - Category (Intensity).

2. **Merging & Resampling Strategy**
To synchronize the sensors, the Accelerometer (12.5Hz) and Gyroscope (25Hz) data were concatenated. A resampling strategy was applied to unify the frequencies:
   - Frequency: Converted to 5Hz (one sample every 200ms).
   - Aggregation Method: Mean value was used for sensor axes to smooth out noise, while the "Last" value was kept for categorical labels.
Benefit: This reduced temporal misalignment and created a computationally efficient dataset.

3. **Dataset Statistics**
   After processing, the final unified dataset consists of 9,009 rows and 10 columns.

   A. Class Distribution (Target Balance)
      The dataset shows a healthy balance across all 6 activity classes:
      - Exercise | Samples
      - OHP (Overhead Press) | 1,676
      - Bench Press | 1,665
      - Squat | 1,610
      - Deadlift | 1,531
      - Row | 1,417
      - Rest | 1,110

   B. Participant Distribution
      Data is spread across 5 participants, ensuring the model generalizes well beyond a single individual:
      - Participant A: 2,988 samples
      - Participant E: 2,645 samples
      - Participant C: 1,481 samples
      - Participant D: 1,052 samples
      - Participant B: 843 samples

4. **Data Quality Assurance**
   A final integrity check confirmed that the dataset is clean and ready for modeling:
      - Missing Values (NaNs): 0 across all columns.
      - Feature Set: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, participant, label, category, set.

---

## **Part 3: Data Visualization**

1. **The Anatomy of a Repetition (Exercise Signatures)**   
By plotting the Accelerometer data, we can observe that every dynamic exercise leaves a distinct 'kinetic signature' in the form of repeating waves (peaks and valleys). Each peak represents a single complete repetition. These regular wave patterns are exactly what the machine learning model will rely on for pattern recognition. The amplitude and shape of the wave vary significantly depending on the exercise type (e.g., the vertical motion of a Squat versus the horizontal pull of a Row).
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/a86ba727c1940c0e472ef0f5cfe164186c741347/reports/figures/participant%20A%20on%20squat.png)

2. **Intensity Comparison (Heavy vs. Medium Weights)**   
When comparing the same exercise performed with different weights (e.g., Heavy vs. Medium sets), a key insight emerges: the time interval between repetitions in a medium-weight set is shorter, indicating a faster execution speed. Conversely, a heavy set takes longer per repetition (lower frequency) and often exhibits more signal 'noise' due to muscle strain and effort. This variance confirms that our model can potentially predict the 'weight category' by analyzing movement speed and stability.
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/a4b528f755f7da93f2c9f8002d094e3962d4b7a0/reports/figures/heavy_vs_medium_sets_y_acc.png)

3. **Participant Variance (Biomechanical Differences)**   
Plotting the same exercise across multiple participants (A, B, C, D, E) reveals that the signals are not perfectly identical, despite it being the exact same movement. This variance is caused by natural biomechanical differences, such as arm length, execution speed, and individual fitness levels. This finding is crucial; it dictates that our model must be trained on a diverse dataset of multiple participants to ensure true generalization and avoid overfitting to a single person's unique movement style.
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/a4b528f755f7da93f2c9f8002d094e3962d4b7a0/reports/figures/participants%20in%20bench%20set%20(acc%20y).png)

4. **Sensor Roles: Accelerometer vs. Gyroscope**   
"The visualizations clearly demonstrate the distinct roles of each sensor. The Accelerometer (top plot) captures the linear acceleration working against gravity, making it excellent for tracking the range of motion and counting repetitions. In contrast, the Gyroscope (bottom plot) records angular velocity and rotation. In exercises requiring strict wrist or torso stability, the Gyroscope signal remains relatively quiet, whereas it shows strong oscillations if the movement involves rotation. Fusing data from both sensors provides the model with a comprehensive 3D view of the physical activity, significantly boosting classification accuracy."
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/a4b528f755f7da93f2c9f8002d094e3962d4b7a0/reports/figures/Bench%20(D).png)


5. **The Static State (Rest Periods)**   
During 'Rest' periods, the signals across both sensors essentially flatline, demonstrating near-total stability. There is only minimal noise present, which is typically caused by the participant's breathing or slight micro-movements. This stark visual contrast between the chaotic 'Active' states and the flat 'Rest' state ensures that classifying resting periods will be a highly accurate and straightforward task for the model.

![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/a4b528f755f7da93f2c9f8002d094e3962d4b7a0/reports/figures/Rest%20(E).png)

---

## **Part 4: Outlier Detection & Handling Strategy**

1. **Exploring Detection Algorithms**
To ensure our machine learning model learns from true kinetic patterns rather than hardware glitches or noise, we experimented with three distinct outlier detection methodologies: the Interquartile Range (IQR) representing a standard statistical approach, the Local Outlier Factor (LOF) representing a density-based machine learning algorithm, and Chauvenet's Criterion representing a distribution-based statistical method.

2. **The Chosen Method: Chauvenet's Criterion**
After visual and analytical inspection, we selected Chauvenet's Criterion as our primary method. The reasoning is clear: kinetic data for a specific exercise naturally distributes around a biomechanical mean. While IQR proved too aggressive—frequently misclassifying the natural high-acceleration peaks of dynamic exercises as outliers—and LOF struggled with our wave-like periodic signals, Chauvenet's Criterion provided the perfect balance. When applied to each exercise individually, it successfully isolated genuine sensor noise without clipping the valid peaks of the movement.

![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/ecf802211e83e66b6ad53fbb068d91f3cda6e1cf/reports/figures/Chauvenet_ACC_x.png)
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/ecf802211e83e66b6ad53fbb068d91f3cda6e1cf/reports/figures/Chauvenet_gyr_x.png)


3. **Post-Detection Action: Handling the Outliers**
Once the outliers were identified, we took a crucial data science approach: we did not delete the affected rows. Dropping rows would disrupt the continuous flow of time and alter our sensor's steady sampling frequency. Instead, we replaced the anomalous values with NaN (Not a Number). This action preserves the temporal structure and integrity of the dataset, preparing it perfectly for the next phase: mathematical imputation (interpolation) to smoothly fill in the gaps.


---

## **Part 5: Feature Engineering & Data Transformation**

Raw sensor data is often noisy and highly dimensional. To prepare this data for machine learning algorithms, we engineered a robust set of features that capture the true kinematic essence of the exercises—such as speed, direction, and hidden movement patterns.
**Transformation Methodology**
1. **Handling Time Gaps (Data Imputation)**
In the previous step, outliers were replaced with NaN to maintain the data's temporal frequency. To fill these gaps without distorting the time flow we used mathematical interpolation. This method smoothly connects the points before and after the gap accurately reconstructing the natural movement curve.
Comparison before (with NaN gaps) and after (smooth mathematical connection):
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/45ec711f4b7edd5ccd02e679f84118e783999653/reports/figures/sample%20with%20nan.png)
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/45ec711f4b7edd5ccd02e679f84118e783999653/reports/figures/sample%20after%20filling%20nans.png)

2. **Signal Denoising (Butterworth Low-pass Filter)**
Accelerometers typically capture high-frequency "vibrations" caused by minor hand movements or device shakes. Since human weightlifting movements are relatively slow (low frequency), we applied a Butterworth Low-pass filter (with a cutoff frequency of $1.3\text{ Hz}$). This filter successfully eliminated high-frequency noise entirely, leaving a smooth, clean kinematic wave that represents actual muscle movement.
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/45ec711f4b7edd5ccd02e679f84118e783999653/reports/figures/before%20and%20after%20lowpass%20filter.png)

3. **Dimensionality Reduction & Pattern Extraction (PCA)**
To reduce data complexity for the model while retaining essential information, we used the PCA (Principal Component Analysis) algorithm. By analyzing the Explained Variance, we found that compressing the original six axes into just 3 Principal Components was sufficient to capture the vast majority of the kinematic variance in the data.
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/45ec711f4b7edd5ccd02e679f84118e783999653/reports/figures/elbow%20tech%20for%20PCA.png)
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/45ec711f4b7edd5ccd02e679f84118e783999653/reports/figures/PCAs%20plot.png)

4. **Orientation Independent Features (Magnitude)**
One of the biggest challenges with wearable sensors is varying device orientation (e.g., placing the phone upside down in a pocket). To overcome this we calculated the total magnitude of movement across the three dimensions using the Pythagorean theorem ($r = \sqrt{x^2 + y^2 + z^2}$). The result is highly robust features (acc_r and gyr_r) that measure absolute movement intensity completely independent of the device's angle or orientation

5. **Unsupervised Pattern Discovery (K-Means Clustering)**
Before training a supervised model, we used the K-Means algorithm to see if the data naturally grouped into distinct exercise categories based on movement patterns. Using the Elbow Method we determined $k=5$ as the optimal number of clusters. When plotting these clusters in a 3D space, the algorithm proved highly successful at visually separating different exercises based on their kinematic properties.
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/45ec711f4b7edd5ccd02e679f84118e783999653/reports/figures/elbow%20tech%20for%20KMEANS.png)
![image alt](https://github.com/ayman23-ds/ML-Project-Fitness-Tracker/blob/45ec711f4b7edd5ccd02e679f84118e783999653/reports/figures/3D%20visualization%20for%20clustring.png)

**Finalizing the Feature Space**
By the end of this pipeline, the dataset became highly enriched with new predictive features. All these features were appended to the original data, including:The three PCA axes (pca_1, pca_2, pca_3).Absolute movement magnitude (acc_r, gyr_r).Temporal and frequency features (Rolling mean and standard deviation using sliding windows, and Fast Fourier Transform (FFT) features to capture exercise cadence).Cluster Labels (to provide an unsupervised baseline for the model).All these features were integrated into the final dataset, which was saved and exported as 03_data_features.pkl. The dataset is now fully prepared in its most robust form for the Predictive Modeling training phase.

---





