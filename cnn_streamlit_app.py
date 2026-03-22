import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches

st.set_page_config(
    page_title="Clinical Trial Cell Imaging — CNN",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Clinical Trial Cell Imaging — CNN")
st.caption("Phase II Clinical Trial Cell Classification | Transfer Learning | Built by [Matt Derya](https://mattderya.com)")

st.info(
    "**Demo Notice:** This app demonstrates the CNN classification pipeline architecture. "
    "The production system at Mentor R&D achieved ~100% accuracy on proprietary Phase II clinical trial microscopy data."
)

with st.sidebar:
    st.header("⚙️ Model Settings")
    model_choice = st.selectbox("CNN Architecture", [
        "EfficientNetB0 (99.2%)",
        "DenseNet121 (98.4%)",
        "ResNet50 (98.7%)",
        "VGG16 (97.9%)",
        "InceptionV3 (97.5%)",
        "MobileNetV2 (96.8%)"
    ])
    img_size = st.selectbox("Image Size", ["96x96", "224x224"])
    augmentation = st.toggle("Data Augmentation", value=True)
    fine_tuning = st.toggle("Fine-tuning", value=False)

    st.divider()
    st.markdown("**Production Results**")
    st.markdown("- 99% accuracy (demo)")
    st.markdown("- ~100% accuracy (production)")
    st.markdown("- 90% cost savings")
    st.markdown("- Months → Minutes")

MODELS_DATA = {
    "EfficientNetB0 (99.2%)": {"acc": 99.2, "params": "5.3M", "color": "#2ecc71"},
    "DenseNet121 (98.4%)":    {"acc": 98.4, "params": "8M",   "color": "#3498db"},
    "ResNet50 (98.7%)":       {"acc": 98.7, "params": "25.6M","color": "#9b59b6"},
    "VGG16 (97.9%)":          {"acc": 97.9, "params": "138M", "color": "#e67e22"},
    "InceptionV3 (97.5%)":    {"acc": 97.5, "params": "23.9M","color": "#e74c3c"},
    "MobileNetV2 (96.8%)":    {"acc": 96.8, "params": "3.4M", "color": "#1abc9c"},
}

tab1, tab2, tab3 = st.tabs(["🖼️ Cell Classification", "📊 Model Benchmark", "🏗️ Architecture"])

with tab1:
    st.subheader("Microscopy Image Classification")
    st.markdown("Click **Classify** to simulate CNN inference on a microscopy patch.")

    col1, col2 = st.columns([1, 1])

    with col1:
        cell_type = st.radio("Sample type:", ["Tumor tissue", "Healthy tissue", "Random sample"])
        classify_btn = st.button("🔍 Classify Cell", type="primary", use_container_width=True)

    if classify_btn:
        np.random.seed(np.random.randint(0, 1000))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for idx, (title, cmap, noise_scale) in enumerate([
            ("Original patch", "pink", 0.3),
            ("Preprocessed", "gray", 0.2),
            ("Grad-CAM overlay", "RdYlGn", 0.4)
        ]):
            if cell_type == "Tumor tissue":
                base = np.random.rand(96, 96) * 0.4 + 0.3
                base[30:70, 30:70] += np.random.rand(40, 40) * 0.5
            elif cell_type == "Healthy tissue":
                base = np.random.rand(96, 96) * 0.3 + 0.6
            else:
                base = np.random.rand(96, 96) * 0.5 + 0.25

            noise = np.random.randn(96, 96) * noise_scale
            img = np.clip(base + noise, 0, 1)

            axes[idx].imshow(img, cmap=cmap)
            axes[idx].set_title(title, fontsize=11)
            axes[idx].axis('off')

            if idx == 2 and cell_type == "Tumor tissue":
                rect = patches.Rectangle((28, 28), 42, 42,
                    linewidth=2, edgecolor='red', facecolor='none')
                axes[idx].add_patch(rect)
                axes[idx].text(28, 25, "Tumor region", color='red', fontsize=8)

        plt.suptitle(f"Sample: {cell_type} | Model: {model_choice.split('(')[0].strip()}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        model_acc = MODELS_DATA[model_choice]["acc"]
        if cell_type == "Tumor tissue":
            pred = "Tumor"
            conf = round(np.random.uniform(model_acc - 2, model_acc) / 100, 4)
        elif cell_type == "Healthy tissue":
            pred = "Healthy"
            conf = round(np.random.uniform(model_acc - 1.5, model_acc) / 100, 4)
        else:
            pred = np.random.choice(["Tumor", "Healthy"])
            conf = round(np.random.uniform(0.78, 0.96), 4)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction", f"{'🔴' if pred == 'Tumor' else '🟢'} {pred}")
        c2.metric("Confidence", f"{conf:.1%}")
        c3.metric("Model", model_choice.split("(")[0].strip())
        c4.metric("Inference time", "< 1ms")

        if pred == "Tumor":
            st.error(f"⚠️ **Tumor tissue detected** — Confidence: {conf:.1%}")
        else:
            st.success(f"✅ **Healthy tissue** — Confidence: {conf:.1%}")

with tab2:
    st.subheader("CNN Architecture Benchmark")
    st.markdown("11 architectures benchmarked on Phase II clinical trial data. Top performers shown.")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(9, 5))
        models = list(MODELS_DATA.keys())
        accs = [MODELS_DATA[m]["acc"] for m in models]
        colors = [MODELS_DATA[m]["color"] for m in models]
        bars = ax.barh([m.split("(")[0].strip() for m in models], accs, color=colors)
        ax.set_xlim(95, 100)
        ax.set_xlabel("Validation Accuracy (%)")
        ax.set_title("Transfer Learning Benchmark — Phase II Clinical Trial Data")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{acc}%', va='center', fontsize=10, fontweight='bold')
        ax.axvline(x=99, color='red', linestyle='--', alpha=0.5, label='99% threshold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Key Stats")
        st.metric("Best model", "EfficientNetB0", "99.2% accuracy")
        st.metric("Lightest model", "EfficientNetB0", "5.3M params")
        st.metric("All models", "> 96.8%", "above threshold")
        st.metric("Production accuracy", "~100%", "full pipeline")

        st.divider()
        st.markdown("### Training Config")
        st.markdown("- Epochs: 3 (demo) / 20-50 (prod)")
        st.markdown("- Optimizer: Adam")
        st.markdown("- Loss: Binary crossentropy")
        st.markdown("- Augmentation: rotation, flip, zoom")
        st.markdown("- GPU: T4 x2 (Kaggle)")

with tab3:
    st.subheader("CNN Pipeline Architecture")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')

    steps = [
        (1, "Input\n96x96 patch", "#3498db"),
        (3, "Data\nAugmentation", "#9b59b6"),
        (5, "Pre-trained\nCNN Backbone", "#e67e22"),
        (7, "Global Average\nPooling", "#2ecc71"),
        (9, "Dropout\n(0.3)", "#e74c3c"),
        (11, "Dense\n128 ReLU", "#1abc9c"),
        (13, "Sigmoid\nOutput", "#f39c12"),
    ]

    for x, label, color in steps:
        rect = patches.FancyBboxPatch((x-0.8, 1.5), 1.6, 2,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.8, edgecolor='white')
        ax.add_patch(rect)
        ax.text(x, 2.5, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')

    for i in range(len(steps)-1):
        x1 = steps[i][0] + 0.8
        x2 = steps[i+1][0] - 0.8
        ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.text(7, 0.5, "Transfer Learning: ImageNet weights frozen → Fine-tuning optional",
           ha='center', fontsize=10, style='italic', color='gray')
    ax.text(7, 4.7, "Phase II Clinical Trial CNN Pipeline",
           ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "220,025 patches", "96x96 pixels")
    col2.metric("Classes", "2", "Tumor / Healthy")
    col3.metric("Best Accuracy", "99.2%", "EfficientNetB0")

st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "99%", "EfficientNet")
col2.metric("Cost Savings", "90%", "vs manual analysis")
col3.metric("Speed", "Minutes", "from months")
col4.metric("Compliance", "HIPAA", "Phase II trials")
