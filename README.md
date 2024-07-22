# Robust Multimodel Fitting with Homography and Fundamental Matrices

This project focuses on enhancing the AdelaideRMF dataset for robust multimodel fitting by cleaning the dataset and converting it into a soft clustering format. The goal is to improve the dataset's quality, making it more suitable for validating multimodel fitting solutions. The project involves methods for inlier/outlier detection and model fitting using robust statistical approaches, particularly focusing on homography and fundamental matrix estimation.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results](#results)
- [Contributors](#contributors)

## Introduction

The AdelaideRMF dataset is widely used for validating multimodel fitting solutions. However, it contains errors and is presented in a hard clustering format, which is not ideal for multimodel fitting scenarios. Our project aims to correct these issues by:
- Cleaning the dataset from errors
- Re-proposing it in the form of soft clustering

This project utilizes robust estimation methods like GC-RANSAC and LMEDS to enhance the dataset and validate the results using clustering performance indexes such as Silhouette scores and Influence functions.

## Project Structure
#### CODE
- `stats.py`: Contains statistical functions for data analysis.
- `visual.py`: Contains functions for data visualization.
- `utils.py`: Utility functions used throughout the project.
- `Inlier_Thresholder.py`: Script for thresholding inliers using change point detection methods.
- `FINAL_H.ipynb`: Jupyter notebook for homography estimation.
- `FINAL_FM.ipynb`: Jupyter notebook for fundamental matrix estimation.
- `Statistics on Inlier Threshold `: Jupyter notebook for statistics on inlier thresholds on Homography case.
- `IACV_PROJECT_SLIDES.pdf`: Presentation slides summarizing the project.
- `IACV_project.pdf`: Detailed project report.
#### DATASET
- `adelFM`: contains the files for the Fundamental Matrix case.
- `adelH`: contains the files for the Homography case.

## Datasets

### AdelaideRMF Dataset

The dataset consists of pairs of images with point correspondences, originally provided in a hard clustering format. Our goal is to convert this into a soft clustering format suitable for multimodel fitting.

## Methodology

### Homography Estimation

1. Apply LMEDS to obtain residuals.
2. Use change point detection on LMEDS residuals to estimate inlier thresholds.
3. Apply GC-RANSAC with the estimated inlier thresholds.
4. Validate outliers using the Influence Function.
5. Create a new dataset with corrected outliers.

### Fundamental Matrix Estimation

1. Compute residuals using Sampson and Symmetric Epipolar Distance.
2. Use change point detection to estimate inlier thresholds for both distance measures.
3. Ensemble models to improve outlier detection.
4. Validate outliers using the Influence Function.
5. Create a new dataset with corrected outliers for significant cases.

### Change Point Detection

Various statistical methods are used to detect changes in the distribution of residuals to determine inlier thresholds. Methods include:
- Inter-Quantile-Range
- Median Absolute Value
- Variance Based
- Estimator Sn and Qn
- Forward Search

### Performance Indexes

- **Silhouette Score**: Measures the quality of clustering.
- **Influence Function**: Evaluates the impact of removing points from the dataset to identify outliers.

## Results

The results demonstrate improved accuracy in identifying and correcting outliers, validated through performance metrics such as Silhouette scores and Influence functions. The enhanced AdelaideRMF dataset provides a better benchmark for multimodel fitting solutions.



# Contributors
Carlo Fabrizio - carlo.fabrizio@mail.polimi.it
Gabriele Giusti - gabriele.giusti@mail.polimi.it
