import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    page_icon="👔",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = keras.models.load_model('fashion_model.h5')
    return model

@st.cache_data
def load_labels():
    with open('outputs/json/labels.json', 'r') as f:
        labels = json.load(f)
    return {int(k): v for k, v in labels.items()}

@st.cache_data
def load_classification_report():
    with open('outputs/json/classification_report.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_test_metrics():
    with open('outputs/json/test_metrics.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_training_history():
    with open('outputs/json/training_history.json', 'r') as f:
        return json.load(f)

def preprocess_image(image):
    img = image.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array, img

def main():
    st.markdown('<h1 class="main-header">👔 Fashion-MNIST Image Classifier</h1>', 
                unsafe_allow_html=True)
    
    try:
        model = load_model()
        labels = load_labels()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("Go to", 
                            ["🔮 Predict", "📈 Model Performance"])
    
    if page == "🔮 Predict":
        st.header("Upload an Image for Classification")
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=['png', 'jpg', 'jpeg', 'bmp'],
                help="Upload a fashion item image (grayscale preferred)"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)
                if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                    with st.spinner('Analyzing image...'):
                        processed_img, processed_pil = preprocess_image(image)
                        predictions = model.predict(processed_img)
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class] * 100
                        with col2:
                            st.subheader("Prediction Results")
                            st.image(processed_pil, caption='Preprocessed (28x28)', 
                                   width=200)
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2 style='color: #1f77b4; margin: 0;'>
                                    {labels[predicted_class]}
                                </h2>
                                <p style='font-size: 1.5rem; margin: 0.5rem 0;'>
                                    Confidence: <strong>{confidence:.2f}%</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.subheader("Confidence for All Classes")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            class_names = [labels[i] for i in range(len(labels))]
                            colors = ['#1f77b4' if i == predicted_class else '#d3d3d3' 
                                    for i in range(len(labels))]
                            bars = ax.barh(class_names, predictions[0] * 100, color=colors)
                            ax.set_xlabel('Confidence (%)', fontsize=12)
                            ax.set_title('Prediction Confidence Distribution', 
                                       fontsize=14, weight='bold')
                            ax.set_xlim([0, 100])

                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                      f'{predictions[0][i]*100:.1f}%',
                                      ha='left', va='center', fontsize=9)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
            else:
                with col2:
                    st.info("Upload an image to get started!")
                    st.markdown("""
                    ### Supported Categories:
                    - T-shirt/top
                    - Trouser
                    - Pullover
                    - Dress
                    - Coat
                    - Sandal
                    - Shirt
                    - Sneaker
                    - Bag
                    - Ankle Boot
                    """)
    elif page == "📈 Model Performance":
        st.header("Model Performance Metrics")
        test_metrics = load_test_metrics()
        classification_report = load_classification_report()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Accuracy", f"{test_metrics['test_accuracy']*100:.2f}%")
        with col2:
            st.metric("Test Loss", f"{test_metrics['test_loss']:.4f}")
        
        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Training History", 
            "🎯 Confusion Matrix", 
            "📋 Classification Report",
            "🖼️ Sample Predictions"
        ])
        with tab1:
            st.subheader("Training vs Validation Curves")
            try:
                img = Image.open('outputs/re_train_model.png')
                st.image(img, use_container_width=True)
            except:
                st.warning("Training history image not found")

            with st.expander("📊 View Raw Training Data"):
                history = load_training_history()
                import pandas as pd
                df = pd.DataFrame(history)
                df.index.name = 'Epoch'
                df.index = df.index + 1
                st.dataframe(df, use_container_width=True)
        with tab2:
            st.subheader("Confusion Matrix")
            try:
                img = Image.open('outputs/confusion_matrix.png')
                st.image(img, use_container_width=True)
            except:
                st.warning("Confusion matrix image not found")
        with tab3:
            st.subheader("Precision, Recall, F1-Score by Class")
            import pandas as pd
            report_data = []
            for class_name, metrics in classification_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_data.append({
                        'Class': class_name,
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1-score']:.4f}",
                        'Support': metrics['support']
                    })
            
            df_report = pd.DataFrame(report_data)
            st.dataframe(df_report, use_container_width=True, hide_index=True)
            st.markdown("### Overall Metrics")
            col1, col2, col3 = st.columns(3)
            
            if 'accuracy' in classification_report:
                with col1:
                    st.metric("Accuracy", 
                            f"{classification_report['accuracy']:.4f}")
            
            if 'macro avg' in classification_report:
                with col2:
                    st.metric("Macro Avg F1-Score", 
                            f"{classification_report['macro avg']['f1-score']:.4f}")
            
            if 'weighted avg' in classification_report:
                with col3:
                    st.metric("Weighted Avg F1-Score", 
                            f"{classification_report['weighted avg']['f1-score']:.4f}")
        with tab4:
            st.subheader("Sample Predictions from Test Set")
            try:
                img = Image.open('outputs/sample_predictions.png')
                st.image(img, use_container_width=True)
                st.caption("Green titles = Correct predictions | Red titles = Incorrect predictions")
            except:
                st.warning("Sample predictions image not found")

if __name__ == "__main__":
    main()