import os
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_URL = (
    "https://huggingface.co/Gastic0712/AI-Wastewise-Model/"
    "resolve/main/efficientnetv2b0_256.keras"
)

MODEL_PATH = "efficientnetv2b0_256.keras"


# =========================================================
# CLASS NAMES
# =========================================================

classes = [
    "aerosol_cans",
    "aluminum_food_cans",
    "aluminum_soda_cans",
    "cardboard_boxes",
    "cardboard_packaging",
    "clothing",
    "coffee_grounds",
    "disposable_plastic_cutlery",
    "eggshells",
    "food_waste",
    "glass_beverage_bottles",
    "glass_cosmetic_containers",
    "glass_food_jars",
    "magazines",
    "newspaper",
    "office_paper",
    "paper_cups",
    "plastic_cup_lids",
    "plastic_detergent_bottles",
    "plastic_food_containers",
    "plastic_shopping_bags",
    "plastic_soda_bottles",
    "plastic_straws",
    "plastic_trash_bags",
    "plastic_water_bottles",
    "shoes",
    "steel_food_cans",
    "styrofoam_cups",
    "styrofoam_food_containers",
    "tea_bags"
]


# =========================================================
# RAG KNOWLEDGE BASE
# =========================================================

knowledge = {

    "steel_food_cans": (
        "Steel/metal",
        "Empty and rinse, then place in recyclable metal waste.",
        "Reuse for storage or planters.",
        "Rinsing removes residue and improves recyclability."
    ),

    "aluminum_food_cans": (
        "Aluminum",
        "Rinse and place with recyclable metal waste.",
        "Reuse for crafts or storage.",
        "Aluminum can be recycled repeatedly."
    ),

    "aluminum_soda_cans": (
        "Aluminum",
        "Empty, rinse and place in recyclable metal waste.",
        "Reuse for crafts.",
        "Avoid putting aluminum cans in general waste."
    ),

    "plastic_water_bottles": (
        "PET plastic",
        "Empty and place in appropriate plastic recycling.",
        "Reuse temporarily for non-food purposes.",
        "Recycling is preferable to single-use disposal."
    ),

    "plastic_soda_bottles": (
        "PET plastic",
        "Empty and place in plastic recycling.",
        "Reuse for storage or crafts.",
        "Keep bottles separate from contaminated waste."
    ),

    "plastic_food_containers": (
        "Plastic",
        "Empty and clean before plastic recycling.",
        "Reuse for storage when appropriate.",
        "Heavily contaminated plastic may not be recyclable."
    ),

    "cardboard_boxes": (
        "Cardboard",
        "Flatten and place with dry paper/cardboard recycling.",
        "Reuse for packaging and storage.",
        "Keep cardboard dry and clean."
    ),

    "cardboard_packaging": (
        "Paper/cardboard",
        "Place clean, dry packaging with paper/cardboard recycling.",
        "Reuse as packaging material.",
        "Remove food contamination before recycling."
    ),

    "glass_beverage_bottles": (
        "Glass",
        "Place in appropriate glass recycling.",
        "Reuse as containers or decorative items.",
        "Handle broken glass carefully."
    ),

    "glass_food_jars": (
        "Glass",
        "Empty and rinse before glass recycling.",
        "Reuse for household storage.",
        "Separate lids if required locally."
    ),

    "paper_cups": (
        "Paper with possible plastic coating",
        "Check local rules for coated cups.",
        "Generally single-use.",
        "Do not assume all paper cups are recyclable."
    ),

    "plastic_trash_bags": (
        "Plastic film",
        "Use plastic-film collection where available.",
        "Reuse for waste collection.",
        "Keep recyclable materials separate."
    ),

    "plastic_straws": (
        "Plastic",
        "Follow local plastic-waste guidelines.",
        "Generally difficult to reuse safely.",
        "Consider reusable alternatives."
    ),

    "shoes": (
        "Mixed materials",
        "Donate or use footwear/textile collection where available.",
        "Donate or repair if usable.",
        "Avoid sending usable footwear directly to landfill."
    ),

    "food_waste": (
        "Organic matter",
        "Place in organic/compost waste where available.",
        "Compost suitable food scraps.",
        "Keep organic waste separate from recyclables."
    ),

    "coffee_grounds": (
        "Organic matter",
        "Add to compost or organic waste where permitted.",
        "Compost or use as soil amendment.",
        "Keep separate from dry recyclables."
    ),

    "eggshells": (
        "Organic/mineral material",
        "Place in compost/organic waste where supported.",
        "Crush and add to compost.",
        "Clean shells before reuse."
    ),

    "tea_bags": (
        "Organic material plus possible synthetic components",
        "Check bag material before composting.",
        "Compost tea leaves when suitable.",
        "Some tea bags contain plastic fibers."
    ),

    "disposable_plastic_cutlery": (
        "Plastic",
        "Follow local plastic-waste guidelines.",
        "Limited reuse only when appropriate.",
        "Reusable cutlery reduces plastic waste."
    ),

    "styrofoam_cups": (
        "Expanded polystyrene",
        "Check whether local facilities accept polystyrene.",
        "Generally single-use.",
        "Check local collection rules."
    ),

    "styrofoam_food_containers": (
        "Expanded polystyrene",
        "Check local recycling/disposal facilities.",
        "Generally single-use.",
        "Reduce use where reusable containers are available."
    )
}


# =========================================================
# DOWNLOAD MODEL
# =========================================================

@st.cache_resource
def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading AI model..."):

            response = requests.get(
                MODEL_URL,
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            with open(MODEL_PATH, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

    return MODEL_PATH


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = download_model()

    return tf.keras.models.load_model(model_path)


# =========================================================
# BUILD GRAD-CAM BACKBONE
# =========================================================

@st.cache_resource
def build_gradcam():

    trained_model = load_model()

    old_backbone = next(
        layer
        for layer in trained_model.layers
        if isinstance(layer, tf.keras.Model)
    )

    backbone = tf.keras.applications.EfficientNetV2B0(
        input_shape=(256, 256, 3),
        include_top=False,
        weights=None
    )

    backbone.set_weights(
        old_backbone.get_weights()
    )

    dense_layer = next(
        layer
        for layer in trained_model.layers[::-1]
        if isinstance(layer, tf.keras.layers.Dense)
    )

    return backbone, dense_layer


# =========================================================
# PREDICTION + GRAD-CAM
# =========================================================

def predict_and_gradcam(image):

    backbone, dense_layer = build_gradcam()

    image_array = tf.keras.utils.img_to_array(image)

    x = tf.expand_dims(image_array, 0)

    with tf.GradientTape() as tape:

        features = backbone(
            x,
            training=False
        )

        tape.watch(features)

        pooled = tf.reduce_mean(
            features,
            axis=[1, 2]
        )

        logits = dense_layer(pooled)

        class_id = tf.argmax(
            logits[0]
        )

        score = logits[:, class_id]

    gradients = tape.gradient(
        score,
        features
    )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        weights[:, None, None, :] * features,
        axis=-1
    )

    cam = tf.maximum(
        cam[0],
        0
    )

    cam = cam / (
        tf.reduce_max(cam) + 1e-8
    )

    cam = tf.image.resize(
        cam[..., None],
        (256, 256)
    ).numpy().squeeze()

    probabilities = tf.nn.softmax(
        logits
    )[0].numpy()

    confidence = float(
        probabilities[int(class_id)]
    )

    return (
        int(class_id),
        confidence,
        cam
    )


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI WasteWise",
    page_icon="♻️",
    layout="centered"
)


# =========================================================
# HEADER
# =========================================================

st.title("♻️ AI WasteWise")

st.write(
    "AI-powered waste classification, "
    "explainability and sustainable waste-management guidance."
)

st.divider()


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is not None:

    image = tf.keras.utils.load_img(
        uploaded_file,
        target_size=(256, 256)
    )

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze Waste",
        use_container_width=True
    ):

        with st.spinner(
            "AI WasteWise is analyzing the image..."
        ):

            class_id, confidence, cam = (
                predict_and_gradcam(image)
            )

        label = classes[class_id]

        information = knowledge.get(
            label,
            (
                "Unknown",
                "Check local waste-management guidelines.",
                "Consider safe reuse where appropriate.",
                "Follow local disposal rules."
            )
        )

        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------

        st.success(
            f"Detected Waste: "
            f"{label.replace('_', ' ').title()}"
        )

        st.metric(
            "AI Confidence",
            f"{confidence:.2%}"
        )

        # ---------------------------------------------
        # GRAD-CAM
        # ---------------------------------------------

        st.subheader(
            "🔥 Explainable AI — Grad-CAM"
        )

        figure, axis = plt.subplots(
            figsize=(7, 7)
        )

        axis.imshow(image)

        axis.imshow(
            cam,
            cmap="jet",
            alpha=0.45
        )

        axis.axis("off")

        axis.set_title(
            f"{label.replace('_', ' ').title()} "
            f"| {confidence:.2%}"
        )

        st.pyplot(
            figure,
            use_container_width=True
        )

        plt.close(figure)

        # ---------------------------------------------
        # RAG GUIDANCE
        # ---------------------------------------------

        st.subheader(
            "♻️ RAG-Based Sustainability Guidance"
        )

        st.write(
            f"**Material:** {information[0]}"
        )

        st.write(
            f"**Disposal:** {information[1]}"
        )

        st.write(
            f"**Reuse:** {information[2]}"
        )

        st.write(
            f"**Sustainability Tip:** {information[3]}"
        )

        # ---------------------------------------------
        # SYSTEM PIPELINE
        # ---------------------------------------------

        st.divider()

        st.subheader(
            "🤖 AI WasteWise Pipeline"
        )

        st.write(
            "📷 Image → "
            "🧠 EfficientNetV2B0 → "
            "🏷️ Classification → "
            "🔥 Grad-CAM → "
            "📚 RAG → "
            "♻️ Sustainability Guidance"
        )

        st.info(
            "This guidance is informational. "
            "Always follow local waste-management "
            "rules for final disposal decisions."
        )

else:

    st.info(
        "👆 Upload a JPG, JPEG or PNG waste image "
        "to start the analysis."
    )
