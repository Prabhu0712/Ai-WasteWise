import os
import io
import json
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from google import genai
from google.genai import types


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI WasteWise",
    page_icon="♻️",
    layout="centered"
)


# =========================================================
# MODEL CONFIG
# =========================================================

MODEL_URL = (
    "https://huggingface.co/Gastic0712/AI-Wastewise-Model/"
    "resolve/main/efficientnetv2b0_256.keras"
)

MODEL_PATH = "efficientnetv2b0_256.keras"


# =========================================================
# 30 TRAINED CLASSES
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

    "Electronic Waste": {
        "material": "Electronics, plastics, metals and electronic components",
        "disposal": (
            "Take the item to an authorized e-waste collection "
            "or electronics recycling facility."
        ),
        "reuse": (
            "Repair, refurbish, donate or reuse the device "
            "if it is still functional."
        ),
        "tip": (
            "Do not dispose of electronic devices with normal "
            "household waste."
        )
    },

    "Plastic": {
        "material": "Plastic",
        "disposal": (
            "Clean the item where appropriate and place it in "
            "the correct plastic recycling stream if accepted locally."
        ),
        "reuse": (
            "Reuse suitable containers and products before disposal."
        ),
        "tip": (
            "Reduce single-use plastics and prefer reusable alternatives."
        )
    },

    "Paper": {
        "material": "Paper",
        "disposal": (
            "Keep the material clean and dry and place it in "
            "paper recycling where available."
        ),
        "reuse": (
            "Reuse paper for notes, packaging or craft purposes."
        ),
        "tip": (
            "Avoid mixing wet or food-contaminated paper with "
            "clean recyclable paper."
        )
    },

    "Cardboard": {
        "material": "Cardboard",
        "disposal": (
            "Flatten clean and dry cardboard before placing it "
            "in cardboard recycling."
        ),
        "reuse": (
            "Reuse boxes for storage, shipping or organization."
        ),
        "tip": "Keep cardboard dry and free from food contamination."
    },

    "Glass": {
        "material": "Glass",
        "disposal": (
            "Place suitable glass items in the appropriate "
            "glass recycling collection."
        ),
        "reuse": (
            "Jars and bottles can sometimes be reused for storage."
        ),
        "tip": "Handle broken glass carefully."
    },

    "Metal": {
        "material": "Metal",
        "disposal": (
            "Clean the item where appropriate and place it in "
            "metal recycling."
        ),
        "reuse": (
            "Reuse suitable containers or send scrap metal "
            "for recycling."
        ),
        "tip": (
            "Metals can often be recovered and recycled instead "
            "of being sent to landfill."
        )
    },

    "Organic Waste": {
        "material": "Biodegradable organic material",
        "disposal": (
            "Use composting or an organic-waste collection "
            "system where available."
        ),
        "reuse": (
            "Suitable organic material can be composted."
        ),
        "tip": (
            "Separating organic waste reduces contamination "
            "of recyclable materials."
        )
    },

    "Textile": {
        "material": "Fabric or mixed textile material",
        "disposal": (
            "Use textile recycling or donation programs where available."
        ),
        "reuse": (
            "Repair, donate or repurpose usable clothing and textiles."
        ),
        "tip": "Extend the useful life of textiles before disposal."
    },

    "Hazardous Waste": {
        "material": "Potentially hazardous material",
        "disposal": (
            "Use an authorized hazardous-waste collection facility "
            "and follow local regulations."
        ),
        "reuse": "Do not reuse hazardous material unless specifically safe.",
        "tip": (
            "Do not mix hazardous waste with ordinary household recycling."
        )
    },

    "General Waste": {
        "material": "Mixed or non-recyclable material",
        "disposal": (
            "Follow local municipal guidelines for residual/general waste."
        ),
        "reuse": (
            "Consider repair or reuse before disposal where appropriate."
        ),
        "tip": (
            "Check whether individual components can be separated "
            "for recycling."
        )
    },

    "Other": {
        "material": "Uncertain or mixed material",
        "disposal": (
            "Check local waste-management guidelines before disposal."
        ),
        "reuse": (
            "Consider repair, donation or reuse where safe."
        ),
        "tip": (
            "Identify the material before choosing a disposal stream."
        )
    }
}


# =========================================================
# MAP TRAINED CLASSES TO BROAD CATEGORIES
# =========================================================

broad_mapping = {

    "aerosol_cans": "Metal",
    "aluminum_food_cans": "Metal",
    "aluminum_soda_cans": "Metal",
    "steel_food_cans": "Metal",

    "cardboard_boxes": "Cardboard",
    "cardboard_packaging": "Cardboard",

    "clothing": "Textile",
    "shoes": "Textile",

    "coffee_grounds": "Organic Waste",
    "eggshells": "Organic Waste",
    "food_waste": "Organic Waste",
    "tea_bags": "Organic Waste",

    "glass_beverage_bottles": "Glass",
    "glass_cosmetic_containers": "Glass",
    "glass_food_jars": "Glass",

    "magazines": "Paper",
    "newspaper": "Paper",
    "office_paper": "Paper",
    "paper_cups": "Paper",

    "disposable_plastic_cutlery": "Plastic",
    "plastic_cup_lids": "Plastic",
    "plastic_detergent_bottles": "Plastic",
    "plastic_food_containers": "Plastic",
    "plastic_shopping_bags": "Plastic",
    "plastic_soda_bottles": "Plastic",
    "plastic_straws": "Plastic",
    "plastic_trash_bags": "Plastic",
    "plastic_water_bottles": "Plastic",

    "styrofoam_cups": "Plastic",
    "styrofoam_food_containers": "Plastic"
}


# =========================================================
# DOWNLOAD + LOAD EFFICIENTNET MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

    return tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# BUILD GRAD-CAM
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
# CUSTOM MODEL PREDICTION + GRAD-CAM
# =========================================================

def efficientnet_prediction(image):

    backbone, dense_layer = build_gradcam()

    resized = image.resize((256, 256))

    image_array = np.array(
        resized,
        dtype=np.float32
    )

    x = tf.expand_dims(
        image_array,
        axis=0
    )

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

        output = dense_layer(pooled)

        class_id = tf.argmax(
            output[0]
        )

        score = output[:, class_id]

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

    # Dense already uses softmax in your trained model.
    probs = output[0].numpy()

    class_id = int(class_id)

    confidence = float(
        probs[class_id]
    )

    return (
        classes[class_id],
        confidence,
        cam
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

@st.cache_resource
def get_gemini_client():

    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        None
    )

    if not api_key:
        return None

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# GEMINI VISUAL WASTE ANALYSIS
# =========================================================

def gemini_analysis(image):

    client = get_gemini_client()

    if client is None:
        return None

    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=90
    )

    image_bytes = buffer.getvalue()

    prompt = """
You are the visual waste-classification component of AI WasteWise.

Analyze the MAIN PHYSICAL OBJECT visible in this image.

Do not force the object into a recycling category if it does not belong there.

For example:
- keyboard, mouse, charger, laptop, phone, circuit board or electronic device
  should be Electronic Waste.
- food scraps should be Organic Waste.
- bottles should be categorized based on their visible material.
- clothing should be Textile.

Allowed broad waste categories are EXACTLY:

Electronic Waste
Plastic
Paper
Cardboard
Glass
Metal
Organic Waste
Textile
Hazardous Waste
General Waste
Other

Return ONLY valid JSON with exactly these fields:

{
  "object_name": "specific object visible",
  "category": "one allowed broad category",
  "material": "main material or materials",
  "reason": "one short explanation",
  "disposal": "short safe disposal recommendation",
  "reuse": "short reuse or repair recommendation",
  "sustainability_tip": "one short sustainability tip"
}

Do not use markdown.
Do not add text before or after the JSON.
If the object cannot be identified reliably, use category "Other".
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )

    text = response.text.strip()

    return json.loads(text)


# =========================================================
# FINAL DECISION LOGIC
# =========================================================

def choose_final_result(
    ml_label,
    ml_confidence,
    gemini_result
):

    ml_category = broad_mapping.get(
        ml_label,
        "Other"
    )

    # Gemini unavailable
    if gemini_result is None:

        return {
            "source": "EfficientNetV2B0",
            "object_name": ml_label.replace("_", " ").title(),
            "category": ml_category,
            "detailed_class": ml_label,
            "confidence": ml_confidence
        }

    gemini_category = gemini_result.get(
        "category",
        "Other"
    )

    gemini_object = gemini_result.get(
        "object_name",
        "Unknown object"
    )

    # -------------------------------------------------
    # Gemini detects something outside our 30 classes
    # -------------------------------------------------

    if gemini_category in [
        "Electronic Waste",
        "Hazardous Waste",
        "General Waste",
        "Other"
    ]:

        return {
            "source": "Gemini Vision",
            "object_name": gemini_object,
            "category": gemini_category,
            "detailed_class": None,
            "confidence": None
        }

    # -------------------------------------------------
    # Both systems agree on broad waste category
    # -------------------------------------------------

    if gemini_category == ml_category:

        return {
            "source": "Hybrid AI",
            "object_name": gemini_object,
            "category": gemini_category,
            "detailed_class": ml_label,
            "confidence": ml_confidence
        }

    # -------------------------------------------------
    # They disagree
    #
    # Gemini gets priority for open-world object
    # identification. EfficientNet result is still shown.
    # -------------------------------------------------

    return {
        "source": "Gemini Vision",
        "object_name": gemini_object,
        "category": gemini_category,
        "detailed_class": None,
        "confidence": None
    }


# =========================================================
# HEADER
# =========================================================

st.title("♻️ AI WasteWise")

st.write(
    "Hybrid AI waste identification using "
    "EfficientNetV2B0, Gemini Vision, Grad-CAM and RAG."
)

st.caption(
    "Upload an object or waste image to identify its "
    "waste category and receive disposal guidance."
)

st.divider()


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📷 Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Analyze Waste",
        type="primary",
        use_container_width=True
    ):

        # ---------------------------------------------
        # CUSTOM MODEL
        # ---------------------------------------------

        with st.spinner(
            "Running AI WasteWise analysis..."
        ):

            ml_label, ml_confidence, cam = (
                efficientnet_prediction(image)
            )

            # -----------------------------------------
            # GEMINI
            # -----------------------------------------

            gemini_result = None

            try:

                gemini_result = gemini_analysis(
                    image
                )

            except Exception as e:

                st.warning(
                    "Gemini visual verification was unavailable. "
                    "The custom classifier result will still be shown."
                )

            # -----------------------------------------
            # FINAL DECISION
            # -----------------------------------------

            final = choose_final_result(
                ml_label,
                ml_confidence,
                gemini_result
            )

        # =============================================
        # FINAL RESULT
        # =============================================

        st.success(
            f"♻️ Waste Category: "
            f"{final['category']}"
        )

        st.subheader(
            final["object_name"]
        )

        st.caption(
            f"Final identification source: "
            f"{final['source']}"
        )

        # =============================================
        # CUSTOM MODEL RESULT
        # =============================================

        with st.expander(
            "🧠 Custom EfficientNetV2B0 Analysis"
        ):

            st.write(
                "**30-Class Prediction:**",
                ml_label.replace(
                    "_",
                    " "
                ).title()
            )

            st.write(
                "**Model Confidence:** "
                f"{ml_confidence:.2%}"
            )

            st.write(
                "**Broad Category:**",
                broad_mapping.get(
                    ml_label,
                    "Other"
                )
            )

            st.caption(
                "The custom model was trained specifically "
                "on 30 household and recyclable waste classes."
            )

        # =============================================
        # GEMINI RESULT
        # =============================================

        if gemini_result:

            with st.expander(
                "✨ Gemini Visual Verification",
                expanded=True
            ):

                st.write(
                    "**Object:**",
                    gemini_result.get(
                        "object_name",
                        "Unknown"
                    )
                )

                st.write(
                    "**Category:**",
                    gemini_result.get(
                        "category",
                        "Other"
                    )
                )

                st.write(
                    "**Material:**",
                    gemini_result.get(
                        "material",
                        "Unknown"
                    )
                )

                st.write(
                    "**Why:**",
                    gemini_result.get(
                        "reason",
                        ""
                    )
                )

        # =============================================
        # GRAD-CAM
        # =============================================

        st.subheader(
            "🔥 Explainable AI — Grad-CAM"
        )

        resized_display = image.resize(
            (256, 256)
        )

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        ax.imshow(
            resized_display
        )

        ax.imshow(
            cam,
            cmap="jet",
            alpha=0.42
        )

        ax.axis(
            "off"
        )

        ax.set_title(
            f"EfficientNet: "
            f"{ml_label.replace('_', ' ').title()}"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(
            fig
        )

        st.caption(
            "Grad-CAM explains the custom EfficientNet "
            "classifier. If Gemini overrides the classification, "
            "the heatmap still corresponds to the EfficientNet result."
        )

        # =============================================
        # GUIDANCE
        # =============================================

        st.subheader(
            "🌱 Sustainability Guidance"
        )

        category = final[
            "category"
        ]

        base_info = knowledge.get(
            category,
            knowledge["Other"]
        )

        # Prefer Gemini's object-specific guidance
        # when available.

        if gemini_result:

            material = gemini_result.get(
                "material",
                base_info["material"]
            )

            disposal = gemini_result.get(
                "disposal",
                base_info["disposal"]
            )

            reuse = gemini_result.get(
                "reuse",
                base_info["reuse"]
            )

            tip = gemini_result.get(
                "sustainability_tip",
                base_info["tip"]
            )

        else:

            material = base_info[
                "material"
            ]

            disposal = base_info[
                "disposal"
            ]

            reuse = base_info[
                "reuse"
            ]

            tip = base_info[
                "tip"
            ]

        st.write(
            f"**Material:** {material}"
        )

        st.write(
            f"**🗑️ Disposal:** {disposal}"
        )

        st.write(
            f"**♻️ Reuse / Repair:** {reuse}"
        )

        st.write(
            f"**🌍 Sustainability Tip:** {tip}"
        )

        # =============================================
        # PIPELINE
        # =============================================

        st.divider()

        st.subheader(
            "🤖 AI WasteWise Pipeline"
        )

        st.write(
            "📷 Image → "
            "🧠 EfficientNetV2B0 → "
            "✨ Gemini Visual Verification → "
            "🔥 Grad-CAM → "
            "📚 RAG Knowledge → "
            "♻️ Waste Guidance"
        )

        st.info(
            "AI predictions and disposal guidance may not "
            "always be correct. Follow your local waste-management "
            "and e-waste regulations for final disposal."
        )


else:

    st.info(
        "👆 Upload a waste or household-object image "
        "to begin."
    )
