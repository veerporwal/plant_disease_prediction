from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import io

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('plant_disease.h5')

# Define class labels
class_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Define class descriptions
class_info = {
    'Corn-Common_rust': 'Common rust is a fungal disease that affects corn plants.',
    'Potato-Early_blight': 'Early blight is a fungal disease that affects potato plants.',
    'Tomato-Bacterial_spot': 'Bacterial spot is a common disease affecting tomato plants.'
}

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the form
        image_file = request.files['image']
        
        # Read the image file
        img = image.load_img(io.BytesIO(image_file.read()), target_size=(256, 256))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Pass the image to the model for prediction
        prediction = model.predict(img_array)
        
        # Get the predicted class label indices
        predicted_classes = np.argmax(prediction, axis=1)
        
        # Get the corresponding class labels and descriptions
        predicted_results = [{'label': class_labels[idx], 'description': class_info[class_labels[idx]]} for idx in predicted_classes]
        
        # Render the result template with the predicted class labels and descriptions
        return render_template('result.html', predicted_results=predicted_results)

if __name__ == '__main__':
    app.run(debug=True)
