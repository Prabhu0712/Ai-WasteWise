import os, io, json, requests
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="AI WasteWise", page_icon="♻️", layout="centered")

MODEL_URL = "https://huggingface.co/Gastic0712/AI-Wastewise-Model/resolve/main/efficientnetv2b0_256.keras"
MODEL_PATH = "efficientnetv2b0_256.keras"
GEMINI_MODEL = "gemini-3.8-flash"

classes = [
    "aerosol_cans","aluminum_food_cans","aluminum_soda_cans","cardboard_boxes",
    "cardboard_packaging","clothing","coffee_grounds","disposable_plastic_cutlery",
    "eggshells","food_waste","glass_beverage_bottles","glass_cosmetic_containers",
    "glass_food_jars","magazines","newspaper","office_paper","paper_cups",
    "plastic_cup_lids","plastic_detergent_bottles","plastic_food_containers",
    "plastic_shopping_bags","plastic_soda_bottles","plastic_straws",
    "plastic_trash_bags","plastic_water_bottles","shoes","steel_food_cans",
    "styrofoam_cups","styrofoam_food_containers","tea_bags"
]

broad = {
    "aerosol_cans":"Metal","aluminum_food_cans":"Metal","aluminum_soda_cans":"Metal",
    "steel_food_cans":"Metal","cardboard_boxes":"Cardboard","cardboard_packaging":"Cardboard",
    "clothing":"Textile","shoes":"Textile","coffee_grounds":"Organic Waste",
    "eggshells":"Organic Waste","food_waste":"Organic Waste","tea_bags":"Organic Waste",
    "glass_beverage_bottles":"Glass","glass_cosmetic_containers":"Glass",
    "glass_food_jars":"Glass","magazines":"Paper","newspaper":"Paper",
    "office_paper":"Paper","paper_cups":"Paper","disposable_plastic_cutlery":"Plastic",
    "plastic_cup_lids":"Plastic","plastic_detergent_bottles":"Plastic",
    "plastic_food_containers":"Plastic","plastic_shopping_bags":"Plastic",
    "plastic_soda_bottles":"Plastic","plastic_straws":"Plastic",
    "plastic_trash_bags":"Plastic","plastic_water_bottles":"Plastic",
    "styrofoam_cups":"Plastic","styrofoam_food_containers":"Plastic"
}

knowledge = {
"Electronic Waste":("Electronics, plastics, metals and electronic components",
"Take it to an authorized e-waste collection or electronics recycling facility.",
"Repair, refurbish, donate or reuse it if functional.",
"Do not dispose of electronic devices with normal household waste."),
"Plastic":("Plastic","Clean where appropriate and use the correct plastic recycling stream.",
"Reuse suitable products before disposal.","Reduce single-use plastics."),
"Paper":("Paper","Keep clean and dry and use paper recycling where available.",
"Reuse for notes, packaging or crafts.","Keep wet and contaminated paper separate."),
"Cardboard":("Cardboard","Flatten clean and dry cardboard for recycling.",
"Reuse boxes for storage or shipping.","Keep cardboard dry."),
"Glass":("Glass","Use the appropriate glass recycling collection.",
"Reuse suitable jars and bottles.","Handle broken glass carefully."),
"Metal":("Metal","Clean where appropriate and use metal recycling.",
"Reuse suitable containers or recycle scrap metal.","Metals can often be recovered and recycled."),
"Organic Waste":("Biodegradable organic material","Use composting or organic-waste collection where available.",
"Compost suitable organic material.","Keep organic waste separate from recyclables."),
"Textile":("Fabric or mixed textile material","Use textile recycling or donation programs.",
"Repair, donate or repurpose textiles.","Extend textile life before disposal."),
"Hazardous Waste":("Potentially hazardous material","Use an authorized hazardous-waste facility.",
"Do not reuse unless specifically safe.","Do not mix hazardous waste with ordinary recycling."),
"General Waste":("Mixed or non-recyclable material","Follow local municipal residual-waste guidelines.",
"Consider repair or reuse first.","Separate recyclable components where possible."),
"Other":("Uncertain, mixed or construction material","Check local waste-management guidelines and specialized collection requirements.",
"Reuse or repurpose suitable material where safe.","Identify the material before choosing disposal.")
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL, stream=True, timeout=300)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def build_gradcam():
    trained = load_model()
    old_backbone = next(x for x in trained.layers if isinstance(x, tf.keras.Model))
    backbone = tf.keras.applications.EfficientNetV2B0(
        input_shape=(256,256,3), include_top=False, weights=None)
    backbone.set_weights(old_backbone.get_weights())
    dense = next(x for x in trained.layers[::-1] if isinstance(x, tf.keras.layers.Dense))
    return backbone, dense

def efficientnet_prediction(image):
    backbone, dense = build_gradcam()
    arr = np.asarray(image.resize((256,256)), dtype=np.float32)
    x = tf.expand_dims(arr, 0)
    with tf.GradientTape() as tape:
        features = backbone(x, training=False)
        tape.watch(features)
        pooled = tf.reduce_mean(features, axis=[1,2])
        output = dense(pooled)
        cid = tf.argmax(output[0])
        score = output[:, cid]
    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1,2))
    cam = tf.reduce_sum(weights[:,None,None,:] * features, axis=-1)
    cam = tf.maximum(cam[0], 0)
    cam /= tf.reduce_max(cam) + 1e-8
    cam = tf.image.resize(cam[...,None], (256,256)).numpy().squeeze()
    probs = output[0].numpy()
    return classes[int(cid)], float(probs[int(cid)]), cam

@st.cache_resource
def gemini_client():
    key = st.secrets.get("GEMINI_API_KEY", None)
    return genai.Client(api_key=key) if key else None

def gemini_analysis(image):
    client = gemini_client()
    if client is None:
        return None

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)

    prompt = """
You are the PRIMARY visual identification system for AI WasteWise.
Identify the MAIN physical object or material visible in the image.
Do NOT force it into the original 30-class taxonomy.

Examples:
keyboard, mouse, laptop, phone, charger, circuit board -> Electronic Waste
concrete, bricks, stones, rubble, masonry, construction debris -> Other
plastic bottle -> Plastic
glass bottle -> Glass
newspaper -> Paper
cardboard box -> Cardboard
banana peel -> Organic Waste
shoe -> Textile
metal can -> Metal

Allowed categories ONLY:
Electronic Waste, Plastic, Paper, Cardboard, Glass, Metal,
Organic Waste, Textile, Hazardous Waste, General Waste, Other.

Return ONLY JSON:
{
"object_name":"specific object or material",
"category":"one allowed category",
"material":"main material",
"reason":"short visual reason",
"disposal":"short disposal recommendation",
"reuse":"short reuse or repair recommendation",
"sustainability_tip":"short sustainability tip"
}

If uncertain, use category Other. Do not add markdown or text outside JSON.
"""

    for _ in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            result = json.loads(response.text.strip())
            allowed = {"Electronic Waste","Plastic","Paper","Cardboard","Glass",
                       "Metal","Organic Waste","Textile","Hazardous Waste",
                       "General Waste","Other"}
            if result.get("category") not in allowed:
                result["category"] = "Other"
            return result
        except Exception:
            pass
    return None

st.title("♻️ AI WasteWise")
st.write("Hybrid AI waste identification using Gemini Vision, EfficientNetV2B0, Grad-CAM and RAG.")
st.caption("Gemini is the primary open-world identifier. EfficientNetV2B0 provides supporting 30-class ML analysis.")
st.divider()

uploaded = st.file_uploader("📷 Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Waste", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            ml_label, ml_conf, cam = efficientnet_prediction(image)
            gemini = gemini_analysis(image)

        if gemini:
            category = gemini.get("category","Other")
            obj = gemini.get("object_name","Unknown object")
            source = "Gemini Vision"
        else:
            category = broad.get(ml_label,"Other")
            obj = ml_label.replace("_"," ").title()
            source = "EfficientNetV2B0 fallback"
            st.warning("Gemini Vision was unavailable. Showing the custom ML fallback.")

        st.success(f"♻️ Waste Category: {category}")
        st.subheader(obj)
        st.caption(f"Final identification source: {source}")

        if gemini:
            with st.expander("✨ Gemini Visual Verification", expanded=True):
                st.write(f"**Object:** {gemini.get('object_name','Unknown')}")
                st.write(f"**Category:** {gemini.get('category','Other')}")
                st.write(f"**Material:** {gemini.get('material','Unknown')}")
                st.write(f"**Why:** {gemini.get('reason','')}")

        with st.expander("🧠 Supporting EfficientNetV2B0 Analysis"):
            st.write(f"**30-Class Prediction:** {ml_label.replace('_',' ').title()}")
            st.write(f"**Model Confidence:** {ml_conf:.2%}")
            st.write(f"**Broad Category:** {broad.get(ml_label,'Other')}")
            st.caption("This is the independent custom 30-class model prediction and does not override Gemini.")

            st.subheader("🔥 Grad-CAM")
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(image.resize((256,256)))
            ax.imshow(cam, cmap="jet", alpha=0.42)
            ax.axis("off")
            ax.set_title("EfficientNetV2B0 attention")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Grad-CAM explains the EfficientNet prediction only, not the Gemini result.")

        info = knowledge.get(category, knowledge["Other"])
        if gemini:
            material = gemini.get("material", info[0])
            disposal = gemini.get("disposal", info[1])
            reuse = gemini.get("reuse", info[2])
            tip = gemini.get("sustainability_tip", info[3])
        else:
            material, disposal, reuse, tip = info

        st.subheader("🌱 Sustainability Guidance")
        st.write(f"**Material:** {material}")
        st.write(f"**🗑️ Disposal:** {disposal}")
        st.write(f"**♻️ Reuse / Repair:** {reuse}")
        st.write(f"**🌍 Sustainability Tip:** {tip}")

        st.divider()
        st.subheader("🤖 AI WasteWise Pipeline")
        st.write("📷 Image → ✨ Gemini Vision → 🧠 EfficientNetV2B0 → 🔥 Grad-CAM → 📚 RAG → ♻️ Guidance")
        st.info("AI guidance is informational. Follow local waste-management and e-waste regulations for final disposal.")
else:
    st.info("👆 Upload a waste or household-object image to begin.")
