import logging
from logging import StreamHandler, FileHandler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. BASIC LOGGING CONFIG
# ---------------------------------------------------------

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------
# 2. CUSTOM LOGGER
# ---------------------------------------------------------

logger = logging.getLogger("iris_ml_app")

# ---------------------------------------------------------
# 3. ADD CONSOLE + FILE HANDLERS
# ---------------------------------------------------------

console_handler = StreamHandler()
file_handler = FileHandler("iris_app.log")

logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------
# 4. LOAD DATA
# ---------------------------------------------------------

logger.info("Loading the Iris dataset...")
try:
    iris = load_iris()
    X = iris.data
    y = iris.target
    logger.debug(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
except Exception:
    logger.exception("Failed to load the Iris dataset")


# ---------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ---------------------------------------------------------

logger.info("Splitting dataset...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.debug(f"Training size: {X_train.shape}, Test size: {X_test.shape}")
except Exception:
    logger.exception("Error during train-test split")


# ---------------------------------------------------------
# 6. DATA PREPROCESSING
# ---------------------------------------------------------

logger.info("Normalizing data with StandardScaler...")
try:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logger.debug("Normalization complete.")
except Exception:
    logger.exception("Error while normalizing data")


# ---------------------------------------------------------
# 7. MODEL TRAINING
# ---------------------------------------------------------

logger.info("Training Logistic Regression model...")
try:
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
except Exception:
    logger.exception("Error during model training")


# ---------------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------------

logger.info("Evaluating the model...")
try:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Model accuracy: {acc:.4f}")

    if acc < 0.7:
        logger.warning("Accuracy is lower than expected.")
except Exception:
    logger.exception("Error during evaluation")


logger.info("Program completed successfully with no errors.")
