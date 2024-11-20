from fastapi import FastAPI, File, UploadFile, Depends
from sqlalchemy.orm import Session
from app.database import Base, engine, SessionLocal
from app.models import TrainingMetadata
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML API"}

@app.post("/train")
async def train_model(file: UploadFile = File(...), db: Session = Depends(get_db)):
    import os

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Save the uploaded file temporarily
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Load the dataset
    try:
        data = pd.read_csv(file_location)
    except Exception as e:
        return {"error": f"Failed to read CSV file: {str(e)}"}

    # Validate the dataset columns
    if 'Y' not in data.columns:
        return {"error": "Dataset must contain 'Y' column as the target"}
    
    # Separate features (all columns except 'Y') and target ('Y')
    try:
        features = data.drop(columns=['Y'])
        target = data['Y']

        # Ensure all feature columns are numeric
        if not all(pd.api.types.is_numeric_dtype(features[col]) for col in features.columns):
            return {"error": "All feature columns must contain numeric data"}
        
        X = features.values  # Features as a numeric array
        y = target.astype(float).values  # Target as a numeric array
    except Exception as e:
        return {"error": f"Data preprocessing error: {str(e)}"}
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate multiple models
    models = {
        "SVM": SVC(kernel="linear", random_state=1),
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=1),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=1)
    }
    
    accuracies = {}
    best_model_name = None
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

        # Update best model if current model is better
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    # Save the best model to disk
    model_path = "data/trained_model.pkl"
    joblib.dump(best_model, model_path)
    
    # Save metadata to the database
    metadata = TrainingMetadata(
        filename=file.filename,
        model_type=best_model_name,
        accuracy=best_accuracy
    )
    db.add(metadata)
    db.commit()
    db.refresh(metadata)

    return {
        "message": f"Best model '{best_model_name}' trained and saved as '{model_path}'",
        "accuracies": accuracies,
        "best_model": best_model_name,
        "best_accuracy": best_accuracy
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import os

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Save the uploaded file temporarily
    file_location = f"data/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Load the dataset
    try:
        data = pd.read_csv(file_location)
    except Exception as e:
        return {"error": f"Failed to read CSV file: {str(e)}"}

    # Check if 'Y' column exists
    has_target = 'Y' in data.columns
    if has_target:
        # Separate features and target
        features = data.drop(columns=['Y'])
        target = data['Y']
    else:
        # Use the entire dataset as features if 'Y' is not present
        features = data

    # Ensure all feature columns are numeric
    try:
        if not all(pd.api.types.is_numeric_dtype(features[col]) for col in features.columns):
            return {"error": "All feature columns must contain numeric data"}
        
        X = features.values  # Features as a numeric array
    except Exception as e:
        return {"error": f"Data preprocessing error: {str(e)}"}

    # Load the best-trained model
    model_path = "data/trained_model.pkl"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load the trained model: {str(e)}"}

    # Make predictions
    try:
        predictions = model.predict(X)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # Calculate accuracy if 'Y' column is present
    accuracy = None
    if has_target:
        accuracy = accuracy_score(target, predictions)

    # Save predictions to a file
    predictions_df = pd.DataFrame({
        "Predicted": predictions
    })
    if has_target:
        predictions_df["Actual"] = target.values

    predictions_file = "data/predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)

    # Return the response
    response = {
        "message": f"Predictions completed and saved as '{predictions_file}'",
        "predictions": predictions.tolist()
    }
    if accuracy is not None:
        response["accuracy"] = accuracy

    return response

@app.get("/metadata")
def get_metadata(db: Session = Depends(get_db)):
    metadata = db.query(TrainingMetadata).all()
    return {"metadata": metadata}

