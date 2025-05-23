import streamlit as st
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
import time
import pandas as pd
import emoji
from ultralytics import YOLO
import cv2
import plotly.express as px

# Set page config
st.set_page_config(page_title="Durian Leaf Disease Detector", layout="wide")

@st.cache_resource
def load_yolo():
    return YOLO(r"model\YOLOv8_best.pt")  # Replace with your actual path

# Add to your existing model loading section
if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = load_yolo()

# Load TFLite model
if "interpreter" not in st.session_state:
    try:
        st.session_state.interpreter = tf.lite.Interpreter(model_path='model/VGG19_model.tflite')
        st.session_state.interpreter.allocate_tensors()
        st.session_state.input_details = st.session_state.interpreter.get_input_details()
        st.session_state.output_details = st.session_state.interpreter.get_output_details()
    except Exception as e:
        st.stop()

interpreter = st.session_state.interpreter
input_details = st.session_state.input_details
output_details = st.session_state.output_details

# Define class names for your model
class_names = ["Algal Leaf Spot", "Leaf Blight", "Leaf Spot", "No Disease"]

# Initialize session_state variables if they do not exist
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []  # To store uploaded images and predictions

# Bolding the Element titles in Sidebar
st.markdown("""
    <style>
    label[data-testid="stWidgetLabel"] p {
        font-weight: bold !important;
        font-size: 18px !important;
    }

    .st-radio label[data-testid="stWidgetLabel"] p {
        font-weight: bold !important;
        font-size: 18px !important;
        padding-top: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Theme and Mode Selection
with st.sidebar:
    mode = st.selectbox(emoji.emojize(":bust_in_silhouette: Select Mode"), ["User", "Developer"], index=0)


    if mode == "Developer":
        pages = ["Home", "Disease Glossary", "Model Interface", "History", "Developer Dashboard"]
    else:
        pages = ["Home", "Disease Glossary", "Model Interface", "History"]

    selected_page = st.radio(
        emoji.emojize(":wrench: Navigate to"),
        pages,
        index=pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0,
        key="page_selector"
    )

    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

# Moving main page content upwards
st.markdown("""
    <style>
    /* Remove top space */
    .main > div:first-child {
        padding-top: 0rem !important;
    }

    /* Optional: adjust padding/margins for the main block to remove any internal spacing */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar alignment adjustment
st.markdown("""
    <style>
    /* Custom Sidebar Styles */
    section[data-testid="stSidebar"] {
        background-color: #DDEEDB !important;
        padding: 0.7rem !important;
        border-right: 2px solid #B6CB83;
        margin-top: -10px !important;  /* Remove the top margin */
        padding-top: 0px !important;  /* Remove top padding */
    }

    /* Ensure text inside the sidebar has proper styling */
    section[data-testid="stSidebar"] * {
        color: #2F3E2E !important;
        font-size: 16px !important;
        line-height: 2 !important;
    }
    
    /* Push down the radio button group */
    div[role="radiogroup"] {
        padding-top: 5px !important; 
    }
    
    div[role="radiogroup"] {
        padding-top: 5px !important; 
    }
    
    div[data-testid="stSelectbox"] {
        margin-bottom: 20px !important; 
    }

    </style>
""", unsafe_allow_html=True)


# Utility Functions
def image_to_base64(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def apply_temperature_scaling(probabilities, temperature):
    logits = np.log(probabilities + 1e-8)
    scaled_logits = logits / temperature
    exp_scaled = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_scaled / np.sum(exp_scaled)

def calculate_brightness(img_array):
    # Convert to grayscale first if needed
    if len(img_array.shape) == 3:
        return np.mean(img_array)
    return np.mean(img_array)


def smart_resize(crop_img, target_size=(224, 224)):
    h, w = crop_img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(crop_img, (new_w, new_h))

    dw = target_size[0] - new_w
    dh = target_size[1] - new_h
    padded = cv2.copyMakeBorder(resized,
                                dh // 2, dh - dh // 2,
                                dw // 2, dw - dw // 2,
                                cv2.BORDER_CONSTANT, value=0)
    return padded

# Page Functions

def show_home():
    st.title(emoji.emojize(":evergreen_tree: Durian Leaf Disease Detector"))

    # Main container with styling
    st.markdown(
        """<div style='border: 2px solid black; border-radius: 10px; padding: 20px; margin-top: 25px; background-color: #f9f9f9; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
        <h3>🔬 Application Overview</h3>
        <p>This web application leverages a machine learning model to detect diseases on durian leaves.</p>
        <p>It is tailored for <strong>farmers, agronomists, and researchers</strong>, enabling early-stage disease detection.</p>
        <p><strong>Diseases Classes predicted by the model:</strong></p>
        <ul><li>Algal Leaf Spot / Bintik Daun Alga</li><li>Leaf Blight / Hawar Daun</li><li>Leaf Spot / Bintik Daun</li><li>No Disease (Healthy condition)</li></ul>
        <h3 style="margin-top: 25px;">📝 How to Use the Model?</h3>
        <ol><li>Navigate to the <strong>Model Interface</strong>.</li><li>Upload a clear image of a durian leaf.</li><li>View the predicted class, confidence score, and calibration info.</li><li>Review previous results in the <strong>History</strong> tab.</li></ol>
        <h3 style="margin-top: 30px;">🚀 Get Started Now!</h3>
        <p>Click the button below to jump straight to the model interface and begin detecting diseases.</p>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Create the button inside Streamlit and style it
    if st.button('Go to Model Interface', key='go_to_model_interface'):
        st.session_state.current_page = "Model Interface"
        st.rerun()

        # Apply styling to the button
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s, transform 0.2s;
        }

        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

def show_info():
    st.title(emoji.emojize(":information_source: Disease Glossary"))
    st.write("This glossary provides information on the durian leaf diseases identified by the model, including their symptoms, causes, and visual characteristics.")

    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    diseases = {
        "Algal Leaf Spot / Bintik Daun Alga": {
            "description": "Algal Leaf Spot is a foliar disease caused by parasitic algae, commonly *Cephaleuros virescens*. It affects tropical fruit trees, including durian.",
            "symptoms": [
                "Circular, reddish-brown (rust-like) or orange spots on upper leaf surfaces.",
                "Fuzzy or powdery appearance due to algal growth.",
                "May coalesce into larger patches under humid conditions.",
                "Premature leaf yellowing and drop in severe cases."
            ],
            "causes": [
                "High humidity and prolonged leaf wetness.",
                "Poor air circulation in dense canopies.",
                "Overhead irrigation or frequent rainfall."
            ],
            "treatment": [
                "Improve air circulation through pruning the tree.",
                "Remove infested shoots or leaves to prevent spread",
                "Apply copper-based fungicides with copper hydroxide or copper oxychloride.",
                "Consider algaecides if the symptoms are severe."
            ],
            "images": [
                "images/Algal_Leaf_Spot_49.jpg",
                "images/Algal_Leaf_Spot_57.jpg"
            ],
            "resources": [
                {
                    "label": "Clemson University - Algal Leaf Spot",
                    "url": "https://hgic.clemson.edu/factsheet/algal-leaf-spot/"
                },
                {
                    "label": "Gardenia.net - Algal Leaf Spot",
                    "url": "https://www.gardenia.net/disease/algal-leaf-spot"
                }
            ]
        },
        "Leaf Blight / Hawar Daun": {
            "description": "Leaf Blight is a rapid and severe condition that causes large areas of leaf tissue to die, often due to fungal pathogens such as *Rhizoctonia solani*.",
            "symptoms": [
                "Large, irregular, brown to dark-brown lesions on leaf edges or tips.",
                "Tissue may appear scorched or burnt.",
                "Lesions spread quickly, especially during wet conditions.",
                "Complete leaf collapse in advanced stages."
            ],
            "causes": [
                "High humidity and ideal temperature of 28°C",
                "Fungal infection by the pathogens 'Rhizoctonia solani' or 'Phytophtora palmivora'.",
                "Excessive rainfall or over-irrigation.",
                "Wounds or injuries on leaf surface."
            ],
            "treatment": [
                "Use foliar sprays or soil drenches like Pencycuron or Benomyl",
                "Apply specialized fungicides containing active ingredients such as Azoxystrobin and Difenoconazole.",
                "Do not allow excessive watering or irrigation.",
                "Ensure proper spacing between plants for better air circulation."
            ],
            "images": [
                "images/Leaf_Blight_71.jpg",
                "images/Leaf_Blight_10.jpg"
            ],
            "resources": [
                {
                    "label": "Sataka - Durian Leaf Blight: Causes and Solutions",
                    "url": "https://sataka.com.vn/en/durian-leaf-blight/#1-causes-of-leaf-blight-in-durian"
                },
                {
                    "label": "Department of Agriculture, Sarawak",
                    "url": "https://doa.sarawak.gov.my/web/attachment/show/?docid=WUs5MFdicDZudnNHVnQ4ZVJmUnNGUT09OjpLPx_axT03fPTh1uQitpsE"
                }
            ]
        },
        "Leaf Spot / Bintik Daun": {
            "description": "Leaf Spot is a common fungal or bacterial condition that manifests as discrete lesions on the leaf surface. It can be caused by species such as *Cercospora* or *Colletotrichum*.",
            "symptoms": [
                "Small, round to oval brown or black spots with yellow halos.",
                "Spots may have defined borders or become necrotic centers.",
                "Gradual spread across the leaf surface.",
                "May lead to decreased photosynthesis if widespread."
            ],
            "causes": [
                "Fungal pathogens such as Cercospora or Colletotrichum.",
                "Prolonged rainfall and high humidity.",
                "Insufficient sunlight due to dense leaf canopies.",
                "Overly dense, overcrowded trees with poor air ventilation"
            ],
            "treatment": [
                "Improve leaf sanitation by removing fallen leaves.",
                "Use organic microbial fertilizers like Trichomix-CTD or Trimix-N1",
                "Apply chemical fungicides with ingredients like Mancozeb and Metalaxyl during early stages on both leaf surfaces.",
                "Ensure plants have adequate nutrition with balanced fertilizers like 15-15-15 NPK (Nitrogen-Phosphorus-Potassiun)."
            ],
            "images": [
                "images/Leaf_Spot_8.jpg",
                "images/Leaf_Spot_79.jpg"
            ],
            "resources": [
                {
                    "label": "Sataka VN - Leaf Spot",
                    "url": "https://sataka.com.vn/en/leaf-spot-disease-on-durian-tree/"
                },
                {
                    "label": "YouTube - Durian Pest and Disease Control",
                    "url": "https://www.youtube.com/watch?v=NG2GWPQZa3I"
                }
            ]
        }
    }

    for name, data in diseases.items():
        with st.expander(name):
            st.markdown(
                f"""
                <div style='
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    padding: 17px;
                    background-color: #f2fdf2;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                    margin-bottom: 10px;
                '>
                    <h3 style='margin-top: 0;'>{name}</h3>
                    <p style='margin-bottom: 15px;'>{data["description"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            left, right = st.columns([2.0, 1.0])  # Adjust column ratio for better visual balance

            with left:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"#### {emoji.emojize(':microscope:')} Symptoms")
                for symptom in data["symptoms"]:
                    st.markdown(f"- {symptom}")

                # Add additional space between Symptoms and Causes sections
                st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

                st.markdown(f"#### {emoji.emojize(':warning:')} Causes")
                for cause in data["causes"]:
                    st.markdown(f"- {cause}")

                # Add additional space between Causes and Treatment sections
                st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

                if "treatment" in data:
                    st.markdown(f"#### {emoji.emojize(':seedling:')} Treatment")
                    for treatment in data["treatment"]:
                        st.markdown(f"- {treatment}")

                    # Add additional space between Treatment and Further Reading sections
                    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

                st.markdown("#### 🔗 Further Reading")
                for res in data["resources"]:
                    st.markdown(f"[{res['label']}]({res['url']})")

            with right:
                st.markdown("<br>", unsafe_allow_html=True)
                # Adjust the image size to 250x250 and ensure alignment to the right
                st.image(data["images"][0], caption=f"{name} Example 1", width=250)
                st.image(data["images"][1], caption=f"{name} Example 2", width=250)

    st.markdown("<br>", unsafe_allow_html=True)


def show_model():
    st.title(emoji.emojize(":toolbox: Durian Disease Detection Model"))
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        """
        **Important Notes Before Use:**
        1.  This model is specifically trained on images of **durian leaves only**. Uploading images of other tree parts (e.g., stems, flowers, fruits) will likely result in unreliable predictions.
        2.  This tool only serves as an initial diagnostic aid. Do consult a **local expert or plant pathologist** for definitive diagnosis and treatment recommendations.
        3.  For **best results**, ensure the uploaded image is clear, well-lit, focuses on the affected area, and has no foreign objects in the background.
        4.  The model's accuracy **may vary** depending on the growth stage of the leaf, the severity of the infection, and image quality.
        5.  You may cross-check the prediction with the example images in glossary for validation purposes.
        """
    )

    treatments = {
        "Algal Leaf Spot": "Apply a copper-based fungicide every 2-3 weeks during wet seasons. Use algaecide if symptoms are too severe. Ensure good air circulation and soil irrigation.",
        "Leaf Blight": "Spray Pencycuron/Monceren or other phenylurea fungicide whcih targets the Rhizoctonia patogen. Avoid watering the trees excessively and ensure spacing between trees.",
        "Leaf Spot": "Prune the trees to prevent dense canopies. Use balanced fertilizers like 15-15-15 NPK for nutrition. Ensure adequate spacing between trees.",
    }

    confidence_threshold = 70.0
    temperature = 2.035

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(emoji.emojize(":camera: Choose an image"), type=["jpg", "jpeg", "png"])
    st.markdown("<br>", unsafe_allow_html=True)

    if mode == "Developer":
        temperature = st.slider(emoji.emojize(":fire: Softmax Temperature Scaling for Model Confidence Calibration"),
                                min_value=1.0, max_value=5.0, value=2.0, step=0.1)

    if uploaded_file:
        start_time = time.time()
        original_image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(original_image)

        # YOLO Detection
        results = st.session_state.yolo_model(img_array)

        # Initialize default values
        use_whole_image = False
        display_image = original_image
        crop_img = img_array

        # Check for detections
        if len(results[0].boxes) > 0:
            # Get detection with highest confidence
            best_box = max(results[0].boxes, key=lambda x: x.conf[0])

            if best_box.conf[0] >= 0.4:  # Only use crop if confidence ≥ 40%
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu().numpy())

                # Crop and validate ROI
                try:
                    crop_img = img_array[y1:y2, x1:x2]
                    if crop_img.size == 0:
                        raise ValueError("Empty crop detected")

                    # Create annotated version
                    annotated_image = results[0].plot()
                    display_image = Image.fromarray(annotated_image)

                except Exception as e:
                    st.error(f"Error processing detected leaf: {str(e)}")
                    return
            else:
                use_whole_image = True
                st.warning("Low confidence in leaf detection - Using entire image for classification")
        else:
            st.error("No leaves detected! Please upload a clearer image of a durian leaf.")
            return

        input_shape = input_details[0]['shape'][1:3]

        if use_whole_image:
            resized_image = Image.fromarray(crop_img).resize(input_shape, resample=Image.BILINEAR)
        else:
            resized_image = Image.fromarray(smart_resize(crop_img, input_shape))

        # Normalize and reshape for inference
        img_array = np.asarray(resized_image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # Copy before slicing for safety
        raw_output = interpreter.get_tensor(output_details[0]['index']).copy()
        output_data = raw_output[0]

        scaled_prediction = apply_temperature_scaling(output_data, temperature)
        predicted_index = int(np.argmax(scaled_prediction))
        predicted_class = class_names[predicted_index]
        confidence_score = float(np.max(scaled_prediction)) * 100

        # Display annotated image instead of original
        base64_image = image_to_base64(display_image)
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 20px;'>
                <img src='data:image/jpeg;base64,{base64_image}'
                     style='width: 500px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); margin-bottom: 10px;' />
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(f"{emoji.emojize(':exploding_head:')} Predicted: **{predicted_class}**")
        st.info(f"{emoji.emojize(':chart_with_downwards_trend:')} Confidence Level: **{confidence_score:.2f}%**")

        if predicted_class != "No Disease" and predicted_class in treatments:
            treatment = treatments.get(predicted_class)
            st.error(f"{emoji.emojize(':pill:')} Suggested Treatment: **{treatment}**")
        elif predicted_class != "No Disease":
            st.warning(f"No specific treatment information available for {predicted_class} at this time.")
        else:
            st.error(f"{emoji.emojize(':pill:')} No treatment prescribed as no diseases were detected.")

        # Keep all your existing UI elements below
        if confidence_score < confidence_threshold:
            st.warning(
                f"{emoji.emojize(':warning:')} The model's confidence in this prediction is only {confidence_score:.2f}%!\n\n"
                f"Please try again with a clearer image of 1 single leaf. If the confidence score is **still low**, either: \n\n"
                f" - The disease/symptoms present could be **different** than the model's trained on.\n\n"
                f" - There are multiple diseases affecting the same leaf. **Check the graph** below for possible diseases\n\n"
                f" - There are foreign objects in the background like plastic bags, hands, machinery, etc.\n\n"
                f"Consult a **local expert** or refer to the **Disease Glossary** for more info {emoji.emojize(':man_farmer:')}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(emoji.emojize(":bar_chart: Confidence Distribution on Possible Diseases"))
        prediction_dict = {class_names[i]: float(scaled_prediction[i]) for i in range(len(class_names))}
        st.bar_chart(prediction_dict)

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "image": base64_image,  # Now stores annotated image
            "predicted_class": predicted_class,
            "confidence_score": confidence_score,
            "temperature": temperature,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            'inference_time': time.time() - start_time,
            'yolo_confidence': best_box.conf[0].item(),
            'image_size': original_image.size
        })

def show_history():
    st.title(emoji.emojize(":eight_o’clock: History"))
    st.write("Here you can see the history of previously uploaded images, predicted classes, and confidence scores.")

    if st.session_state.history:
        for entry in st.session_state.history:
            image_data = base64.b64decode(entry["image"])
            image = Image.open(BytesIO(image_data))
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()

            st.markdown(
                f"""
                <div style='
                    display: flex;
                    border: 2px solid black;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    background-color: #f9f9f9;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                '>
                    <div style='flex: 0 0 200px;'>
                        <img src="data:image/jpeg;base64,{img_b64}" style='width: 100%; border-radius: 8px;' />
                    </div>
                    <div style='flex: 1; padding-left: 20px; display: flex; flex-direction: column; justify-content: center;'>
                        <h4 style='margin: 0;'>🤯 Predicted Class: <span style="color: darkgreen;">{entry["predicted_class"]}</span></h4>
                        <p style='margin: 5px 0 0 0;'>📉 Confidence Score: <strong>{entry["confidence_score"]:.2f}%</strong></p>
                        <p style='margin: 2px 0;'>🧺 Temperature Used: {entry["temperature"]}</p>
                        <p style='margin: 2px 0;'>⏱ Timestamp: {entry["timestamp"]}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.write(emoji.emojize(":memo: No history yet for this session."))

    st.markdown("<br><br>", unsafe_allow_html=True)

def show_dashboard():
    st.title(emoji.emojize(":round_pushpin: Developer Dashboard"))
    st.write("This dashboard provides insights into the model behavior, input quality, and prediction trends.")

    if not st.session_state.history:
        st.warning("No predictions have been made yet.")
        return

    history_df = pd.DataFrame(st.session_state.history)

    # Shows the latest prediction made and details
    st.subheader(emoji.emojize(":pushpin: Latest Prediction"))
    latest = history_df.iloc[-1]
    st.write(f"Predicted Class: {latest['predicted_class']}")
    st.write(f"Confidence Score: {latest['confidence_score']:.2f}%")
    st.write(f"Temperature Used: {latest['temperature']}")
    st.write(f"Timestamp: {latest['timestamp']}")

    # Confidence Distribution Histogram
    st.subheader(emoji.emojize(":input_numbers: Confidence Score Distribution"))
    fig = px.histogram(history_df, x="confidence_score",
                     nbins=20, title="Confidence Score Distribution")
    st.plotly_chart(fig)

    # Class-Wise Confidence Analysis
    st.subheader(emoji.emojize(":bar_chart: Class-Wise Confidence Metrics"))
    class_stats = history_df.groupby('predicted_class')['confidence_score'].agg(['mean', 'std', 'count'])
    st.dataframe(class_stats.style.background_gradient(cmap='Blues'))
    fig = px.bar(class_stats.reset_index(),
                 x='predicted_class',
                 y='mean',
                 error_y='std',
                 title="Average Confidence ± Variability")
    st.plotly_chart(fig)

    # Data Quality Metrics
    st.subheader(emoji.emojize(":frame_with_picture: Input Image Quality Analysis"))
    # Add image quality metrics collection to your prediction flow
    history_df['brightness'] = history_df['image'].apply(
        lambda x: calculate_brightness(np.array(Image.open(BytesIO(base64.b64decode(x))))))
    fig = px.box(history_df, x='predicted_class', y='brightness',
                title="Brightness Distribution by Class")
    st.plotly_chart(fig)

    # Model Latency Monitoring
    st.subheader(emoji.emojize(":stopwatch: Inference Latency Tracking"))
    fig = px.line(history_df, x='timestamp', y='inference_time',
                 title="Prediction Latency Over Time")
    st.plotly_chart(fig)

    # Leaf Detection Quality Analysis
    st.subheader(emoji.emojize(":round_pushpin: YOLOv8 Leaf Detection Confidence Analysis"))
    fig = px.scatter(history_df, x='yolo_confidence', y='confidence_score',
                    color='predicted_class', title="Relationship between YOLOv8 Confidence and VGG-19 Confidence")
    st.plotly_chart(fig)

    # Writes a summary for the local session
    st.subheader(emoji.emojize(":clipboard: Session Summary"))
    st.write(f"Total Predictions: {len(history_df)}")
    st.write(f"Average Confidence: {history_df['confidence_score'].mean():.2f}%")
    st.write(f"Most Common Prediction: {history_df['predicted_class'].mode()[0]}")

    st.markdown("<br>", unsafe_allow_html=True)


# Main logic
def main():
    if st.session_state.current_page == "Home":
        show_home()
    elif st.session_state.current_page == "Disease Glossary":
        show_info()
    elif st.session_state.current_page == "Model Interface":
        show_model()
    elif st.session_state.current_page == "History":
        show_history()
    elif st.session_state.current_page == "Developer Dashboard" and mode == "Developer":
        show_dashboard()
    elif st.session_state.current_page == "Developer Dashboard" and mode != "Developer":
        st.warning("You need to switch to Developer Mode to access the Developer Dashboard.")

# Run the app
main()