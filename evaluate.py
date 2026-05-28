from tensorflow.keras.models import load_model
from data_loader import val_data

# Load trained model
model = load_model("models/plant_disease_model.h5")

# Evaluate model
loss, accuracy = model.evaluate(val_data)

# Print results
print("\nModel Evaluation")
print("Loss:", loss)
print("Accuracy:", accuracy * 100, "%")