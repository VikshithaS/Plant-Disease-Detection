import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_option_menu import option_menu

# Page Config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Theme CSS
st.markdown("""
<style>

/* Main Background */
.stApp{
    background-color:#eef7ee;
}

/* Text */
html, body, p, div, span, label, li{
    color:#1b4332 !important;
}

/* Headings */
h1,h2,h3{
    color:#1b4332 !important;
    font-weight:700;
}

/* Prediction Card */
.prediction-card{
    background-color:#d8f3dc;
    padding:20px;
    border-radius:18px;
    color:#081c15 !important;
    box-shadow:0 4px 12px rgba(0,0,0,0.15);
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#2d6a4f;
}

/* Upload box */
[data-testid="stFileUploader"]{
    background-color:white;
    padding:15px;
    border-radius:12px;
}

/* Upload Button */
[data-testid="stFileUploader"] button{
    background-color:#2d6a4f !important;
    color:white !important;
    border:none !important;
    border-radius:12px !important;
    font-weight:700 !important;
}

[data-testid="stFileUploader"] button *{
    color:white !important;
    fill:white !important;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = load_model(
    "models/plant_disease_model.keras"
)

# Classes
classes = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Solutions
solutions = {

    "Tomato_Early_blight": {
        "precaution": """
- Avoid overwatering
- Remove infected leaves
- Maintain proper airflow
- Keep foliage dry
""",

        "suggestion": """
- Apply fungicide treatment
- Improve ventilation
- Rotate crops regularly
- Monitor nearby plants
"""
    },

    "Potato___Early_blight": {
        "precaution": """
- Avoid excess moisture
- Remove infected leaves
- Maintain field hygiene
""",

        "suggestion": """
- Apply fungicides
- Improve air circulation
"""
    },

    "Potato___Late_blight": {
        "precaution": """
- Keep leaves dry
- Avoid overwatering
""",

        "suggestion": """
- Use copper fungicide
- Remove infected plants
"""
    },

    "Tomato_Late_blight": {
        "precaution": """
- Reduce humidity
- Avoid excessive irrigation
""",

        "suggestion": """
- Apply fungicide
- Remove infected plants
"""
    },

    "Tomato_healthy": {
        "precaution": """
- Continue monitoring
- Maintain irrigation
""",

        "suggestion": """
- Continue balanced care
- Maintain nutrition
"""
    }
}

# Sidebar Navbar
with st.sidebar:

    selected = option_menu(
        menu_title="🌿 Menu",
        options=[
            "Home",
            "Disease Detection",
            "About Project"
        ],
        icons=[
            "house-fill",
            "search",
            "book-fill"
        ],
        menu_icon="leaf-fill",
        default_index=0,
        styles={
            "container": {
                "padding": "10px",
                "background-color": "#2d6a4f"
            },
            "icon": {
                "color": "white",
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "8px",
                "color": "white",
                "--hover-color": "#40916c",
                "border-radius": "10px"
            },
            "nav-link-selected": {
                "background-color": "#40916c",
                "color": "white",
                "border-radius": "10px"
            },
        }
    )

    st.markdown("---")

    st.success(
        "AI-based Plant Disease Detection using CNN + Deep Learning"
    )

# HOME
if selected == "Home":

    st.markdown("""
    <h1 style='text-align:center;'>
    🌿 Plant Disease Detection System
    </h1>
    """, unsafe_allow_html=True)

    st.write("""
Detect plant diseases using AI and Deep Learning.

### Features
- Disease Prediction
- Confidence Score
- Precautions
- Treatment Suggestions
""")

# DETECTION
elif selected == "Disease Detection":

    st.title("🔍 Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload Plant Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:

            img = Image.open(
                uploaded_file
            ).convert("RGB")

            col1, col2 = st.columns(2)

            # LEFT
            with col1:

                st.image(
                    img,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            img_resized = img.resize(
                (224,224)
            )

            img_array = np.array(
                img_resized
            )

            img_array = img_array / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            prediction = model.predict(
                img_array
            )

            predicted_class = classes[
                np.argmax(prediction)
            ]

            confidence = np.max(
                prediction
            ) * 100

            # RIGHT
            with col2:

                if "healthy" in predicted_class.lower():
                    st.success("🌱 Healthy Plant")
                else:
                    st.error("⚠ Disease Detected")

                st.markdown(
                    f"""
                    <div class='prediction-card'>
                    <h3>Predicted Disease</h3>
                    <h2>{predicted_class}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    "### Confidence Level"
                )

                st.progress(
                    int(confidence)
                )

                st.info(
                    f"{confidence:.2f}% Confidence"
                )

                if predicted_class in solutions:

                    st.subheader(
                        "⚠ Precautions"
                    )

                    st.markdown(
                        solutions[predicted_class]["precaution"]
                    )

                    st.subheader(
                        "💡 Suggestions"
                    )

                    st.markdown(
                        solutions[predicted_class]["suggestion"]
                    )

                else:

                    st.warning(
                        "No information available."
                    )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# ABOUT
elif selected == "About Project":

    st.title("📘 About Project")

    st.write("""
This project uses CNN and Deep Learning
to detect plant diseases from leaf images.

### Features
- Disease Classification
- Confidence Prediction
- Precautions
- Treatment Suggestions
- Streamlit Web Application
""")