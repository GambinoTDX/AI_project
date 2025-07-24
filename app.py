import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt




import streamlit as st

# HTML code for the Distracted Driving AI webpage with extensive CSS
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"> <!-- Character encoding for the document -->
    <meta name="viewport" content="width=device-width, initial-scale=1.0"> <!-- Responsive design -->
    <title>Distracted Driving AI</title> <!-- Title of the webpage -->
    <style>
        body {
            font-family: 'Arial', sans-serif; /* Font style for the body */
            background-color: #e9ecef; /* Light grey background */
            margin: 0; /* Remove default margin */
            padding: 20px; /* Add padding to the body */
            color: #343a40; /* Dark text color */
        }
        h1 {
            font-size: 48px; /* Largest font size for the main heading */
            text-align: center; /* Center the heading */
            color: #007bff; /* Blue color for the main heading */
            margin-bottom: 20px; /* Space below the heading */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        h2 {
            font-size: 36px; /* Second largest font size for subheading */
            text-align: center; /* Center the subheading */
            margin-top: 30px; /* Space above the subheading */
            margin-bottom: 10px; /* Space below the subheading */
            color: #6c757d; /* Grey color for the subheading */
        }
        p {
            font-size: 20px; /* Reasonable font size for paragraphs */
            line-height: 1.5; /* Increase line height for readability */
            text-align: center; /* Center align paragraph text */
            margin: 10px 0; /* Margin above and below paragraphs */
        }
        img {
            width: 300px; /* Set a specific width for images */
            margin: 20px; /* Margin around images */
            border-radius: 15px; /* Rounded corners for images */
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); /* Shadow for a lifted effect */
            transition: transform 0.3s; /* Smooth transition for hover effect */
        }
        img:hover {
            transform: scale(1.05); /* Slightly enlarge image on hover */
        }
        footer {
            margin-top: 40px; /* Space above the footer */
            text-align: center; /* Center align footer text */
            font-size: 16px; /* Smaller font size for footer */
            color: #6c757d; /* Grey color for footer text */
        }
    </style>
</head>
<body>
    <h1>Welcome to Distracted Driving AI</h1> <!-- Main heading -->
    <p>This project aims to analyze and detect distracted driving behaviors using AI technology.</p> <!-- Description of the project -->

    <h2>Sample Images</h2> <!-- Subheading for images -->
    <p>Below are some sample images related to distracted driving:</p> <!-- Description for images -->

    <img src="Image 1 that works.jpg" alt="Sample Image 1"> <!-- Placeholder for the first image -->
    <img src="Image 2 that works.jpg" alt="Sample Image 2"> <!-- Placeholder for the second image -->
    

    <footer>
        <p>Feel free to replace the images with your own samples!</p> <!-- Note for users -->
        <p>&copy; 2025 Distracted Driving AI Project</p> <!-- Copyright notice -->
    </footer>
</body>
</html>
"""

# Use Streamlit to render the HTML
st.html(html_code)

# def get_saliency_map(model, input_image, class_index):
#     """
#     generate saliency map for a given class and image
#     """
#     # define loss function
#     score = CategoricalScore(class_index)

#     # create saliency object
#     saliency = Saliency(model,
#                         clone=True)

#     # generate saliency map
#     saliency_map = saliency(score, input_image)
#     saliency_map = normalize(saliency_map)

#     return saliency_map

# def saliency_to_rgb(saliency_map):
#     """
#     convert saliency map to rgb using blue-red color scheme
#     """
#     # normalize the saliency map for better visualization
#     saliency_map_normalized = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

#     saliency_map_colored = plt.get_cmap('jet')(saliency_map_normalized)[:, :, :3]  # exclude alpha channel
#     saliency_map_colored = (saliency_map_colored * 255).astype(np.uint8)  # convert to uint8 for displaying as an image

#     return saliency_map_colored

# def overlay_saliency_map(image, saliency_map, alpha=0.8):
#     """
#     overlay saliency map on image
#     """
#     image = image[0].astype(np.uint8)
#     saliency_map = saliency_map[0]
#     saliency_rgb = saliency_to_rgb(saliency_map)
#     return (image * (1 - alpha) + saliency_rgb * alpha).astype(np.uint8)

model = load_model('./cnn.keras')
def predict_image(img):
    # Preprocess the image
    img = image.resize((64, 64))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Fill in Class Labels (i.e., what are our four classes?)
    class_labels = ['Attentive', 'DrinkingCoffee', 'UsingRadio','UsingMirror']
    return class_labels[predicted_class]

# Create your upload image button
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    # sal = get_saliency_map(model, img_to_array(image.resize((64,64))), ([1]) )
    st.write(predict_image('Uploaded Image'))
    # st.image(saliency_to_rgb(sal), caption="Saliency Map", use_container_width=True)
    # st.image(overlay_saliency_map(img_to_array(image.resize((64,64))), sal), caption="Saliency Map", use_container_width=True)

    # Make your prediction here
    # FILL THIS IN

# YOUR CODE HERE - add to use a saliency map and overlay it!
