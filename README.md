# Iris ML Application with Logging

A demonstration project showcasing the logging implementation in a machine learning pipeline using the Iris dataset. This lab illustrates best practices for logging at different levels (INFO, DEBUG, ERROR) and exception handling in ML workflows.

## Overview

This code trains a Logistic Regression model on the Iris dataset while demonstrating proper logging practices throughout the ML pipeline. It includes two versions of the logger to show both normal execution and exception handling scenarios.

## Features

- **Multi-level Logging**: Implements INFO, DEBUG, WARNING, and ERROR level logging
- **Dual Output**: Logs to both console (StreamHandler) and file (FileHandler)
- **Exception Tracking**: Error handling with detailed tracebacks
- **ML Pipeline Coverage**: Logs every stage from data loading to model evaluation
- **Demonstration Modes**: Includes both clean execution and intentional error scenarios

## Project Structure
```
logger.py             # Main script with clean execution
logger_error.py       # Script with intentional error for demo
iris_app.log          # Log file from normal execution
iris_app_error.log    # Log file with exception demonstration
requirements.txt      # Project dependencies
terminal_output.txt   # Sample console outputs
```

## Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

**Dependencies:**
- scikit-learn==1.5.0
- numpy==1.26.4
- pandas==2.2.2
- matplotlib==3.8.4

## Usage

### Normal Execution

Run the standard logging demonstration:
```bash
python3 logger.py
```

This executes the complete ML pipeline with logging at each stage and completes successfully.

### Exception Demonstration

Run the version with intentional error:
```bash
python3 logger_error.py
```

This demonstrates exception logging with a deliberate ZeroDivisionError to show how errors are captured and logged.

## Pipeline Stages

The application logs the following stages:

1. **Data Loading**: Loads Iris dataset and logs shapes
2. **Train-Test Split**: Creates 80/20 split with logging
3. **Data Preprocessing**: Normalizes features using StandardScaler
4. **Model Training**: Trains Logistic Regression model
5. **Evaluation**: Computes and logs model accuracy
6. **Exception Handling**: Demonstrates error logging (in error version)

## Logging Configuration

The logger is configured with:

- **Logger Name**: `iris_ml_app`
- **Log Level**: DEBUG (captures all log levels)
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Handlers**: 
  - Console output (StreamHandler)
  - File output to `iris_app.log` (FileHandler)

## Sample Output

### Successful Execution
```
Loading the Iris dataset...
2025-11-16 18:36:36,426 - iris_ml_app - INFO - Loading the Iris dataset...
Dataset shape: (150, 4), Labels shape: (150,)
2025-11-16 18:36:36,428 - iris_ml_app - DEBUG - Dataset shape: (150, 4), Labels shape: (150,)
Splitting dataset...
2025-11-16 18:36:36,428 - iris_ml_app - INFO - Splitting dataset...
Training size: (120, 4), Test size: (30, 4)
2025-11-16 18:36:36,429 - iris_ml_app - DEBUG - Training size: (120, 4), Test size: (30, 4)
Normalizing data with StandardScaler...
2025-11-16 18:36:36,429 - iris_ml_app - INFO - Normalizing data with StandardScaler...
Normalization complete.
2025-11-16 18:36:36,430 - iris_ml_app - DEBUG - Normalization complete.
Training Logistic Regression model...
2025-11-16 18:36:36,430 - iris_ml_app - INFO - Training Logistic Regression model...
Model training complete.
2025-11-16 18:36:36,435 - iris_ml_app - INFO - Model training complete.
Evaluating the model...
2025-11-16 18:36:36,435 - iris_ml_app - INFO - Evaluating the model...
Model accuracy: 1.0000
2025-11-16 18:36:36,436 - iris_ml_app - INFO - Model accuracy: 1.0000
Program completed successfully with no errors.
2025-11-16 18:36:36,436 - iris_ml_app - INFO - Program completed successfully with no errors.
```

### Exception Logging
```
(logging) anirudhsayya@Anirudhs-MacBook-Pro logging_lab5 % python3 logger.py 
Loading the Iris dataset...
2025-11-16 18:33:40,974 - iris_ml_app - INFO - Loading the Iris dataset...
Dataset shape: (150, 4), Labels shape: (150,)
2025-11-16 18:33:40,976 - iris_ml_app - DEBUG - Dataset shape: (150, 4), Labels shape: (150,)
Splitting dataset...
2025-11-16 18:33:40,976 - iris_ml_app - INFO - Splitting dataset...
Training size: (120, 4), Test size: (30, 4)
2025-11-16 18:33:40,976 - iris_ml_app - DEBUG - Training size: (120, 4), Test size: (30, 4)
Normalizing data with StandardScaler...
2025-11-16 18:33:40,976 - iris_ml_app - INFO - Normalizing data with StandardScaler...
Normalization complete.
2025-11-16 18:33:40,977 - iris_ml_app - DEBUG - Normalization complete.
Training Logistic Regression model...
2025-11-16 18:33:40,977 - iris_ml_app - INFO - Training Logistic Regression model...
Model training complete.
2025-11-16 18:33:40,983 - iris_ml_app - INFO - Model training complete.
Evaluating model...
2025-11-16 18:33:40,983 - iris_ml_app - INFO - Evaluating model...
Model accuracy: 1.0000
2025-11-16 18:33:40,983 - iris_ml_app - INFO - Model accuracy: 1.0000
Demonstrating exception logging with a deliberate error...
2025-11-16 18:33:40,983 - iris_ml_app - INFO - Demonstrating exception logging with a deliberate error...
An error occurred: division by zero in demo block
Traceback (most recent call last):
  File "/Users/anirudhsayya/Documents/Mylabs_mlops/Github_Labs/Lab5_logging/logging_lab5/logger.py", line 111, in <module>
    faulty_calc = 10 / 0  # will raise ZeroDivisionError
ZeroDivisionError: division by zero
2025-11-16 18:33:40,983 - iris_ml_app - ERROR - An error occurred: division by zero in demo block
Traceback (most recent call last):
  File "/Users/anirudhsayya/Documents/Mylabs_mlops/Github_Labs/Lab5_logging/logging_lab5/logger.py", line 111, in <module>
    faulty_calc = 10 / 0  # will raise ZeroDivisionError
ZeroDivisionError: division by zero
```

## Model Performance

The Logistic Regression model achieves:
- **Accuracy**: 1.0000 (100%) on the test set
- **Dataset**: 150 samples, 4 features, 3 classes
- **Train/Test Split**: 120/30 samples

## Learning Outcomes

This project demonstrates:

- Setting up Python's logging module with custom loggers
- Using multiple handlers for different output destinations
- Implementing appropriate log levels for different scenarios
- Exception logging with full tracebacks
- Logging best practices in ML pipelines
- Debugging and monitoring production ML applications

## Best Practices Shown

1. **Structured Logging**: Clear messages at each pipeline stage
2. **Exception Handling**: Try-except blocks with logger.exception()
3. **Debug Information**: Dataset shapes and intermediate results
4. **Performance Monitoring**: Model accuracy logging with warnings for poor performance
5. **Dual Output**: Console for development, file for production tracking

## Notes

- The logger captures both the raw print output and formatted log entries
- Log files are created in the same directory as the script
- The `logger.exception()` method automatically includes the full traceback
- All ML operations are wrapped in try-except blocks for robust error handling