# main.py
import data_preprocessing
import model_training
import model_evaluation
import utils
import visualizations

# Load and preprocess data
train_data, test_data = data_preprocessing.load_data('mnist_train.csv', 'mnist_test.csv')
X_train, y_train, X_test, y_test = data_preprocessing.preprocess_data(train_data, test_data)

# Define and train the model
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
model = model_training.create_model(input_shape, num_classes)
history = model_training.train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=128)

# Evaluate the model
model_evaluation.evaluate_model(model, X_test, y_test)

# Plot accuracy and loss over epochs
utils.plot_accuracy_loss(history)

# Visualize sample images
visualizations.plot_sample_images(X_test, model.predict_classes(X_test))


