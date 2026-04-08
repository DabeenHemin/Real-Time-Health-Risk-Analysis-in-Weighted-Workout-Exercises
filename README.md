# Workout Health Analysis

## Overview
The aim of this project is to develop a system that combines computer vision and machine learning techniques to analyse an individual's form whilst performing weighted exercises. The system is designed to identify improper techniques that may lead to injury and provide efficient real-time feedback using a webcam.

## Features
- Pose detection using MediaPipe (33 human body landmarks)
- Feature extraction from body landmarks (e.g. joint angles, distances)
- Machine learning classification of exercise form
- Real-time feedback during exercise
- Repetition (rep) counting

## Project Structure
- `src/` → core application scripts (extraction, live system, utilities)
- `data/` → raw and processed datasets
- `models/` → trained machine learning models
- `notebooks/` → model training (Google Colab)

## Status
The project is currently under development. Initial work focuses on squat data extraction and pose analysis, with further development planned for additional exercises such as bicep curls and shoulder presses.