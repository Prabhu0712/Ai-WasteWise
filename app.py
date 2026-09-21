import os
import io
import json
import base64
import requests

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from google import genai
from google.genai import types


# ============================================================
# AI WASTEWISE
#
# PRIMARY:
#   Gemini Vision
#
# SECONDARY:
#   Meta AI / Meta Model API
#
# FALLBACK:
#   EfficientNetV2B0 30-class model
#
# EXPLAINABILITY:
#   Grad-CAM
#
# KNOWLEDGE:
#   RAG sustainability guidance
# ============================================================

st.set_page_config(
    page_title="AI WasteWise",
    page_icon="♻️",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_URL = (
    "https://huggingface.co/Gastic0712/AI-Wastewise-Model/"
    "resolve/main/efficientnetv2b0_256.keras"
)

MODEL_PATH = "efficientnetv2b0_256.keras"

# Gemini
GEMINI_MODEL = "gemini-3.8-flash"

# Meta Model API
META_MODEL = "muse-spark-1.3"
META_API_URL = "https://api.meta.ai/v1/chat/completions"


# ============================================================
# 30 CUSTOM EFFICIENTNET CLASSES
# ============================================================

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
    "tea_bags",
]


# ============================================================
# BROAD CATEGORIES
# ============================================================

broad = {

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
    "styrofoam_food_containers": "Plastic",
}


# ============================================================
# RAG KNOWLEDGE BASE
# ============================================================

knowledge = {

    "Electronic Waste": (
        "Electronics, plastics, metals and electronic components",
        "Take it to an authorized e-waste collection or electronics recycling facility.",
        "Repair, refurbish, donate or reuse it if functional.",
        "Do not dispose of electronic devices with normal household waste."
    ),

    "Plastic": (
        "Plastic",
        "Clean where appropriate and use the correct plastic recycling stream.",
        "Reuse suitable products before disposal.",
        "Reduce single-use plastics."
    ),

    "Paper": (
        "Paper",
        "Keep clean and dry and use paper recycling where available.",
        "Reuse for notes, packaging or crafts.",
        "Keep wet and contaminated paper separate."
    ),

    "Cardboard": (
        "Cardboard",
        "Flatten clean and dry cardboard for recycling.",
        "Reuse boxes for storage or shipping.",
        "Keep cardboard dry."
    ),

    "Glass": (
        "Glass",
        "Use the appropriate glass recycling collection.",
        "Reuse suitable jars and bottles.",
        "Handle broken glass carefully."
    ),

    "Metal": (
        "Metal",
        "Clean where appropriate and use metal recycling.",
        "Reuse suitable containers or recycle scrap metal.",
        "Metals can often be recovered and recycled."
    ),

    "Organic Waste": (
        "Biodegradable organic material",
        "Use composting or organic-waste collection where available.",
        "Compost suitable organic material.",
        "Keep organic waste separate from recyclables."
    ),

    "Textile": (
        "Fabric or mixed textile material",
        "Use textile recycling or donation programs.",
        "Repair, donate or repurpose textiles.",
        "Extend textile life before disposal."
    ),

    "Hazardous Waste": (
        "Potentially hazardous material",
        "Use an authorized hazardous-waste facility.",
        "Do not reuse unless specifically safe.",
        "Do not mix hazardous waste with ordinary recycling."
    ),

    "General Waste": (
        "Mixed or non-recyclable material",
        "Follow local municipal residual-waste guidelines.",
        "Consider repair or reuse first.",
        "Separate recyclable components where possible."
    ),

    "Other": (
        "Uncertain, mixed or construction material",
        "Check local waste-management guidelines and specialized collection requirements.",
        "Reuse or repurpose suitable material where safe.",
        "Identify the material before choosing disposal."
    ),
}


# ============================================================
# AI PROMPT
# ============================================================

AI_PROMPT = """
You are the PRIMARY visual identification system for AI WasteWise.

Identify the MAIN physical object or material visible in the image.

IMPORTANT:
Do NOT force an unknown object into the original 30-class taxonomy.

Identify the actual object/material.

Examples:

keyboard, mouse, laptop, phone, charger, circuit board
-> Electronic Waste

concrete, bricks, stones, rubble, masonry, construction debris
-> Other

plastic bottle/container
-> Plastic

glass bottle/jar
-> Glass

newspaper/paper
-> Paper

cardboard box
-> Cardboard

food scraps, eggshells, tea waste
-> Organic Waste

shoe, clothing, fabric
-> Textile

metal can/object
-> Metal

Allowed categories ONLY:

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

Return ONLY valid JSON.

Use exactly this structure:

{
    "object_name": "specific object or material",
    "category": "one allowed category",
    "material": "main material",
    "reason": "short visual reason",
    "disposal": "short disposal recommendation",
    "reuse": "short reuse or repair recommendation",
    "sustainability_tip": "short sustainability tip"
}

If uncertain, use category "Other".

Do not add markdown.
Do not add text outside JSON.
"""


ALLOWED_CATEGORIES = {
    "Electronic Waste",
    "Plastic",
    "Paper",
    "Cardboard",
    "Glass",
    "Metal",
    "Organic Waste",
    "Textile",
    "Hazardous Waste",
    "General Waste",
    "Other"
}


# ============================================================
# API KEY FUNCTIONS
# ============================================================

def get_secret(name):

    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""

    if not value:
        value = os.getenv(name, "")

    return str(value).strip()


def get_gemini_key():
    return get_secret("GEMINI_API_KEY")


def get_meta_key():
    """
    Supports both names:

    META_AI_API_KEY
    MODEL_API_KEY

    META_AI_API_KEY is preferred.
    """

    key = get_secret("META_AI_API_KEY")

    if not key:
        key = get_secret("MODEL_API_KEY")

    return key


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    key = get_gemini_key()

    if not key:
        return None

    try:
        return genai.Client(
            api_key=key
        )
    except Exception:
        return None


# ============================================================
# EFFICIENTNET MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        response = requests.get(
            MODEL_URL,
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        with open(
            MODEL_PATH,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# GRAD-CAM MODEL
# ============================================================

@st.cache_resource
def build_gradcam():

    trained = load_model()

    old_backbone = next(
        layer
        for layer in trained.layers
        if isinstance(
            layer,
            tf.keras.Model
        )
    )

    backbone = tf.keras.applications.EfficientNetV2B0(

        input_shape=(256, 256, 3),

        include_top=False,

        weights=None
    )

    backbone.set_weights(
        old_backbone.get_weights()
    )

    dense = next(
        layer
        for layer in reversed(
            trained.layers
        )
        if isinstance(
            layer,
            tf.keras.layers.Dense
        )
    )

    return backbone, dense


# ============================================================
# EFFICIENTNET PREDICTION
# ============================================================

def efficientnet_prediction(image):

    backbone, dense = build_gradcam()

    arr = np.asarray(
        image.resize(
            (256, 256)
        ),
        dtype=np.float32
    )

    x = tf.expand_dims(
        arr,
        axis=0
    )

    with tf.GradientTape() as tape:

        features = backbone(
            x,
            training=False
        )

        tape.watch(
            features
        )

        pooled = tf.reduce_mean(
            features,
            axis=[1, 2]
        )

        output = dense(
            pooled
        )

        class_id = tf.argmax(
            output[0],
            output_type=tf.int32
        )

        score = output[
            :,
            class_id
        ]

    gradients = tape.gradient(
        score,
        features
    )

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        weights[:, None, None, :]
        * features,
        axis=-1
    )

    cam = tf.maximum(
        cam[0],
        0
    )

    cam = cam / (
        tf.reduce_max(cam)
        + 1e-8
    )

    cam = tf.image.resize(
        cam[..., None],
        (256, 256)
    ).numpy().squeeze()

    probabilities = output[0].numpy()

    predicted_id = int(
        class_id.numpy()
    )

    return (
        classes[predicted_id],
        float(
            probabilities[predicted_id]
        ),
        cam
    )


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(image):

    client = get_gemini_client()

    if client is None:
        return None

    try:

        buffer = io.BytesIO()

        image.convert(
            "RGB"
        ).save(
            buffer,
            format="JPEG",
            quality=90
        )

        image_bytes = (
            buffer.getvalue()
        )

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=[

                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),

                AI_PROMPT
            ],

            config=types.GenerateContentConfig(

                response_mime_type="application/json"
            )
        )

        text = (
            response.text or ""
        ).strip()

        if not text:
            return None

        result = json.loads(
            text
        )

        category = result.get(
            "category",
            "Other"
        )

        if category not in ALLOWED_CATEGORIES:

            category = "Other"

        result["category"] = category

        defaults = {

            "object_name":
                "Unknown object",

            "material":
                "Unknown",

            "reason":
                "Visual identification.",

            "disposal":
                knowledge["Other"][1],

            "reuse":
                knowledge["Other"][2],

            "sustainability_tip":
                knowledge["Other"][3]
        }

        for key, default in defaults.items():

            if not result.get(key):

                result[key] = default

        return result

    except Exception:

        # IMPORTANT:
        # Do not expose API errors to the user.
        return None


# ============================================================
# META AI / META MODEL API ANALYSIS
# ============================================================

def analyze_with_meta(image):

    api_key = get_meta_key()

    if not api_key:
        return None

    try:

        buffer = io.BytesIO()

        image.convert(
            "RGB"
        ).save(
            buffer,
            format="JPEG",
            quality=90
        )

        image_bytes = (
            buffer.getvalue()
        )

        encoded_image = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        image_data_url = (
            "data:image/jpeg;base64,"
            + encoded_image
        )

        payload = {

            "model": META_MODEL,

            "messages": [

                {
                    "role": "user",

                    "content": [

                        {
                            "type": "text",
                            "text": AI_PROMPT
                        },

                        {
                            "type": "image_url",

                            "image_url": {

                                "url":
                                image_data_url
                            }
                        }
                    ]
                }
            ]
        }

        response = requests.post(

            META_API_URL,

            headers={

                "Authorization":
                f"Bearer {api_key}",

                "Content-Type":
                "application/json"
            },

            json=payload,

            timeout=120
        )

        if response.status_code != 200:

            return None

        data = response.json()

        choices = data.get(
            "choices",
            []
        )

        if not choices:
            return None

        message = choices[0].get(
            "message",
            {}
        )

        text = message.get(
            "content",
            ""
        )

        if not text:
            return None

        text = text.strip()

        # Remove markdown JSON fences if returned.
        if text.startswith(
            "```json"
        ):

            text = text[
                7:
            ].strip()

        if text.startswith(
            "```"
        ):

            text = text[
                3:
            ].strip()

        if text.endswith(
            "```"
        ):

            text = text[
                :-3
            ].strip()

        result = json.loads(
            text
        )

        category = result.get(
            "category",
            "Other"
        )

        if category not in ALLOWED_CATEGORIES:

            category = "Other"

        result["category"] = category

        defaults = {

            "object_name":
                "Unknown object",

            "material":
                "Unknown",

            "reason":
                "Visual identification.",

            "disposal":
                knowledge["Other"][1],

            "reuse":
                knowledge["Other"][2],

            "sustainability_tip":
                knowledge["Other"][3]
        }

        for key, default in defaults.items():

            if not result.get(key):

                result[key] = default

        return result

    except Exception:

        # IMPORTANT:
        # Do not expose Meta API errors to the user.
        return None


# ============================================================
# HYBRID AI ROUTER
# ============================================================

def hybrid_ai_analysis(image):

    """
    Priority:

    1. Gemini
    2. Meta AI
    3. None

    If Gemini key does not exist:
        Meta is tried.

    If Gemini exists but fails:
        Meta is tried.

    If Meta also fails:
        EfficientNet fallback is used.

    No API-key error is shown to the user.
    """

    # --------------------------------------------------------
    # FIRST: GEMINI
    # --------------------------------------------------------

    if get_gemini_key():

        result = analyze_with_gemini(
            image
        )

        if result:

            return (
                result,
                "Gemini Vision"
            )


    # --------------------------------------------------------
    # SECOND: META AI
    # --------------------------------------------------------

    if get_meta_key():

        result = analyze_with_meta(
            image
        )

        if result:

            return (
                result,
                "Meta AI"
            )


    # --------------------------------------------------------
    # NO EXTERNAL AI AVAILABLE
    # --------------------------------------------------------

    return (
        None,
        None
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "♻️ AI WasteWise"
)

st.write(
    "Hybrid AI waste identification using "
    "Gemini Vision, Meta AI, EfficientNetV2B0, "
    "Grad-CAM and RAG."
)

st.caption(
    "AI Vision identifies the object. "
    "EfficientNetV2B0 provides supporting custom ML analysis."
)

st.divider()


uploaded = st.file_uploader(

    "📷 Upload an image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded:

    image = Image.open(
        uploaded
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

        with st.spinner(
            "Analyzing image..."
        ):

            # Custom model
            ml_label, ml_conf, cam = (
                efficientnet_prediction(
                    image
                )
            )

            # External AI router
            ai_result, ai_source = (
                hybrid_ai_analysis(
                    image
                )
            )


        # ====================================================
        # FINAL IDENTIFICATION
        # ====================================================

        if ai_result:

            category = ai_result.get(
                "category",
                "Other"
            )

            object_name = ai_result.get(
                "object_name",
                "Unknown object"
            )

            source = ai_source

        else:

            # Silent fallback.
            category = broad.get(
                ml_label,
                "Other"
            )

            object_name = (
                ml_label
                .replace(
                    "_",
                    " "
                )
                .title()
            )

            source = (
                "EfficientNetV2B0"
            )


        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.success(
            f"♻️ Waste Category: {category}"
        )

        st.subheader(
            object_name
        )

        st.caption(
            f"Final identification source: {source}"
        )


        # ====================================================
        # AI VERIFICATION
        # ====================================================

        if ai_result:

            with st.expander(
                f"✨ {ai_source} Visual Verification",
                expanded=True
            ):

                st.write(
                    f"**Object:** "
                    f"{ai_result['object_name']}"
                )

                st.write(
                    f"**Category:** "
                    f"{ai_result['category']}"
                )

                st.write(
                    f"**Material:** "
                    f"{ai_result['material']}"
                )

                st.write(
                    f"**Why:** "
                    f"{ai_result['reason']}"
                )


        # ====================================================
        # SUPPORTING EFFICIENTNET
        # ====================================================

        with st.expander(
            "🧠 Supporting EfficientNetV2B0 Analysis"
        ):

            st.write(
                "**30-Class Prediction:** "
                + ml_label
                .replace(
                    "_",
                    " "
                )
                .title()
            )

            st.write(
                f"**Model Confidence:** "
                f"{ml_conf:.2%}"
            )

            st.write(
                "**Broad Category:** "
                + broad.get(
                    ml_label,
                    "Other"
                )
            )

            st.caption(
                "This is the independent custom "
                "30-class model prediction. "
                "It does not override the external "
                "AI visual identification."
            )


            # =================================================
            # GRAD-CAM
            # =================================================

            st.subheader(
                "🔥 Grad-CAM"
            )

            fig, ax = plt.subplots(
                figsize=(6, 6)
            )

            ax.imshow(
                image.resize(
                    (256, 256)
                )
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
                "EfficientNetV2B0 attention"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(
                fig
            )

            st.caption(
                "Grad-CAM explains the EfficientNetV2B0 "
                "prediction only."
            )


        # ====================================================
        # SUSTAINABILITY GUIDANCE
        # ====================================================

        st.subheader(
            "🌱 Sustainability Guidance"
        )


        if ai_result:

            info = knowledge.get(
                category,
                knowledge["Other"]
            )

            material = ai_result.get(
                "material",
                info[0]
            )

            disposal = ai_result.get(
                "disposal",
                info[1]
            )

            reuse = ai_result.get(
                "reuse",
                info[2]
            )

            tip = ai_result.get(
                "sustainability_tip",
                info[3]
            )

        else:

            material, disposal, reuse, tip = (
                knowledge.get(
                    category,
                    knowledge["Other"]
                )
            )


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


        # ====================================================
        # PIPELINE
        # ====================================================

        st.divider()

        st.subheader(
            "🤖 AI WasteWise Pipeline"
        )

        if ai_source == "Gemini Vision":

            st.write(
                "📷 Image → "
                "✨ Gemini Vision → "
                "🧠 EfficientNetV2B0 → "
                "🔥 Grad-CAM → "
                "📚 RAG → "
                "♻️ Guidance"
            )

        elif ai_source == "Meta AI":

            st.write(
                "📷 Image → "
                "✨ Meta AI → "
                "🧠 EfficientNetV2B0 → "
                "🔥 Grad-CAM → "
                "📚 RAG → "
                "♻️ Guidance"
            )

        else:

            st.write(
                "📷 Image → "
                "🧠 EfficientNetV2B0 → "
                "🔥 Grad-CAM → "
                "📚 RAG → "
                "♻️ Guidance"
            )


        st.info(
            "AI guidance is informational. "
            "Follow local waste-management and "
            "e-waste regulations for final disposal."
        )


else:

    st.info(
        "👆 Upload a waste or household-object image "
        "to begin."
    )
