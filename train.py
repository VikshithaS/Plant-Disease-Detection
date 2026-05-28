from augmentation import train_data, val_data
from cnn_model import model

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Save model
model.save("models/plant_disease_model.keras")

print("Model trained and saved successfully!")