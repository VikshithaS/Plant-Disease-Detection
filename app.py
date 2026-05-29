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

/* Background */
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
    padding:25px;
    border-radius:18px;
    color:#081c15 !important;
    box-shadow:0 4px 12px rgba(0,0,0,0.15);
    word-wrap:break-word;
    overflow-wrap:break-word;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#2d6a4f;
}

/* Upload Box */
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

# Load Model
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

# Solutions for ALL Classes
solutions = {

    "Pepper__bell___Bacterial_spot": {
        "precaution": """
- Avoid overhead watering
- Remove infected leaves
- Maintain field sanitation
- Avoid working on wet plants
""",
        "suggestion": """
- Apply copper bactericide
- Improve airflow
- Practice crop rotation
- Monitor nearby plants
"""
    },

    "Pepper__bell___healthy": {
        "precaution": """
- Continue regular monitoring
- Maintain proper watering
- Keep soil healthy
""",
        "suggestion": """
- Maintain balanced fertilization
- Continue good plant care
- Monitor for early symptoms
"""
    },

    "Potato___Early_blight": {
        "precaution": """
- Avoid excess moisture
- Remove infected leaves
- Maintain field hygiene
""",
        "suggestion": """
- Apply fungicide
- Improve air circulation
- Rotate crops
"""
    },

    "Potato___Late_blight": {
        "precaution": """
- Keep foliage dry
- Avoid overwatering
- Remove infected plants
""",
        "suggestion": """
- Apply copper fungicide
- Improve drainage
- Monitor disease spread
"""
    },

    "Potato___healthy": {
        "precaution": """
- Continue monitoring
- Maintain irrigation
- Keep soil fertile
""",
        "suggestion": """
- Maintain balanced nutrition
- Regular field inspection
- Follow preventive care
"""
    },

    "Tomato_Bacterial_spot": {
        "precaution": """
- Avoid overhead watering
- Remove infected leaves
- Maintain field hygiene
- Avoid touching wet plants
""",
        "suggestion": """
- Apply copper-based bactericide
- Improve air circulation
- Practice crop rotation
- Monitor nearby plants
"""
    },

    "Tomato_Early_blight": {
        "precaution": """
- Avoid overwatering
- Remove infected leaves
- Maintain airflow
- Keep foliage dry
""",
        "suggestion": """
- Apply fungicide
- Improve ventilation
- Rotate crops
- Monitor disease spread
"""
    },

    "Tomato_Late_blight": {
        "precaution": """
- Reduce humidity
- Avoid excessive irrigation
- Remove diseased leaves
""",
        "suggestion": """
- Apply fungicide
- Remove infected plants
- Improve drainage
"""
    },

    "Tomato_Leaf_Mold": {
        "precaution": """
- Reduce humidity
- Avoid overcrowding
- Maintain ventilation
""",
        "suggestion": """
- Apply fungicide
- Improve airflow
- Remove infected leaves
"""
    },

    "Tomato_Septoria_leaf_spot": {
        "precaution": """
- Avoid leaf wetness
- Remove infected leaves
- Keep field clean
""",
        "suggestion": """
- Apply fungicide
- Practice crop rotation
- Monitor regularly
"""
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "precaution": """
- Maintain plant moisture
- Monitor leaf undersides
- Remove infested leaves
""",
        "suggestion": """
- Apply miticide
- Use neem oil spray
- Introduce beneficial insects
"""
    },

    "Tomato__Target_Spot": {
        "precaution": """
- Avoid excess humidity
- Remove infected foliage
- Improve spacing
""",
        "suggestion": """
- Apply fungicide
- Maintain ventilation
- Rotate crops
"""
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "precaution": """
- Remove infected plants
- Control whitefly population
- Maintain hygiene
- Avoid overcrowding
""",
        "suggestion": """
- Use resistant varieties
- Apply insecticides
- Monitor regularly
- Practice crop rotation
"""
    },

    "Tomato__Tomato_mosaic_virus": {
        "precaution": """
- Avoid touching healthy plants after infected ones
- Remove infected plants
- Sterilize tools
""",
        "suggestion": """
- Use resistant varieties
- Maintain hygiene
- Monitor fields regularly
"""
    },

    "Tomato_healthy": {
        "precaution": """
- Continue monitoring
- Maintain irrigation
- Keep plants healthy
""",
        "suggestion": """
- Maintain balanced fertilization
- Continue preventive care
- Monitor for symptoms
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

# HOME PAGE
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
- User-Friendly Interface
""")

    st.info(
        "Upload a plant leaf image to identify diseases and receive recommendations."
    )

# DISEASE DETECTION PAGE
elif selected == "Disease Detection":

    st.title("🔍 Disease Detection")

    uploaded_file = st.file_uploader(
        "Upload Plant Leaf Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        try:

            img = Image.open(
                uploaded_file
            ).convert(
                "RGB"
            )

            col1, col2 = st.columns(
                2
            )

            # LEFT COLUMN
            with col1:

                st.image(
                    img,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            # Preprocess Image
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

            # Prediction
            prediction = model.predict(
                img_array
            )

            predicted_class = classes[
                np.argmax(
                    prediction
                )
            ]

            confidence = np.max(
                prediction
            ) * 100

            # RIGHT COLUMN
            with col2:

                if "healthy" in predicted_class.lower():

                    st.success(
                        "🌱 Healthy Plant"
                    )

                else:

                    st.error(
                        "⚠ Disease Detected"
                    )

                # Prediction Card
                st.markdown(
                    f"""
                    <div class='prediction-card'>
                    <h3>Predicted Disease</h3>

                    <h2 style='
                    font-size:32px;
                    line-height:1.3;
                    word-wrap:break-word;
                    overflow-wrap:break-word;
                    '>

                    {predicted_class.replace("_"," ")}

                    </h2>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Confidence
                st.write(
                    "### Confidence Level"
                )

                st.progress(
                    int(confidence)
                )

                st.info(
                    f"{confidence:.2f}% Confidence"
                )

                # Precautions & Suggestions
                if predicted_class in solutions:

                    st.subheader(
                        "⚠ Precautions"
                    )

                    st.markdown(
                        solutions[
                            predicted_class
                        ][
                            "precaution"
                        ]
                    )

                    st.subheader(
                        "💡 Suggestions"
                    )

                    st.markdown(
                        solutions[
                            predicted_class
                        ][
                            "suggestion"
                        ]
                    )

                else:

                    st.warning(
                        "No information available."
                    )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# ABOUT PAGE
elif selected == "About Project":

    st.title(
        "📘 About Project"
    )

    st.write("""
This project uses CNN and Deep Learning
to detect plant diseases from leaf images.

### Features
- Disease Classification
- Confidence Prediction
- Precautions
- Treatment Suggestions
- Streamlit Web Application
- AI-powered Agriculture Support
""")

    st.success(
        "Validation Accuracy Achieved: 93.33%"
    )