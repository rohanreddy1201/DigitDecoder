# model_evaluation.py
import numpy as np

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')

def predict_classes(model, X_test):
    y_pred = model.predict(X_test)
    return np.argmax(y_pred, axis=1)
