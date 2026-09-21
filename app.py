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

st.set_page_config(page_title="AI WasteWise", page_icon="♻️", layout="wide")

MODEL_URL = "https://huggingface.co/Gastic0712/AI-Wastewise-Model/resolve/main/efficientnetv2b0_256.keras"
MODEL_PATH = "efficientnetv2b0_256.keras"
GEMINI_MODEL = "gemini-2.5-flash"

classes = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes",
    "cardboard_packaging", "clothing", "coffee_grounds", "disposable_plastic_cutlery",
    "eggshells", "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers",
    "glass_food_jars", "magazines", "newspaper", "office_paper", "paper_cups",
    "plastic_cup_lids", "plastic_detergent_bottles", "plastic_food_containers",
    "plastic_shopping_bags", "plastic_soda_bottles", "plastic_straws",
    "plastic_trash_bags", "plastic_water_bottles", "shoes", "steel_food_cans",
    "styrofoam_cups", "styrofoam_food_containers", "tea_bags"
]

broad = {
    "aerosol_cans": "Metal", "aluminum_food_cans": "Metal", "aluminum_soda_cans": "Metal",
    "steel_food_cans": "Metal", "cardboard_boxes": "Cardboard", "cardboard_packaging": "Cardboard",
    "clothing": "Textile", "shoes": "Textile", "coffee_grounds": "Organic Waste",
    "eggshells": "Organic Waste", "food_waste": "Organic Waste", "tea_bags": "Organic Waste",
    "glass_beverage_bottles": "Glass", "glass_cosmetic_containers": "Glass",
    "glass_food_jars": "Glass", "magazines": "Paper", "newspaper": "Paper",
    "office_paper": "Paper", "paper_cups": "Paper", "disposable_plastic_cutlery": "Plastic",
    "plastic_cup_lids": "Plastic", "plastic_detergent_bottles": "Plastic",
    "plastic_food_containers": "Plastic", "plastic_shopping_bags": "Plastic",
    "plastic_soda_bottles": "Plastic", "plastic_straws": "Plastic",
    "plastic_trash_bags": "Plastic", "plastic_water_bottles": "Plastic",
    "styrofoam_cups": "Plastic", "styrofoam_food_containers": "Plastic"
}

# Regional Segregation Guidelines for RAG
REGIONAL_RULES = {
    "Standard Municipal / CPCB (India)": "Follow Central Pollution Control Board (CPCB) guidelines: Blue Bin (Dry Recyclables), Green Bin (Wet/Biodegradable), Black/Red Bin (Domestic Hazardous). E-waste must be taken exclusively to authorized EPR collection centers.",
    "EU Circular Economy Standard": "Strict separation: Yellow/Blue/Green streams. Plastic packaging in lightweight bins, clean paper/board in blue bins, glass by color, organic bio-waste in dedicated brown bins, WEEE electronics at retail drop-offs.",
    "US / General Curbside Single-Stream": "Clean and dry items only in blue curbside bins (flattened cardboard, rigid plastic containers #1 & #2, aluminum). Soft plastic films, styrofoam, and electronics are strictly forbidden from single-stream curbside bins."
}

KNOWLEDGE_BASE = {
    "Electronic Waste": {
        "material": "Circuit boards, battery compounds, copper, plastics, metals",
        "disposal": "Take strictly to an authorized e-waste collection center or authorized drop-off. Never place in standard curbside bins.",
        "reuse": "Repair, refurbish, donate if functional, or salvage components for spare parts.",
        "tip": "Prevents toxic heavy metals (lead, cadmium, mercury) from leaching into municipal soil and groundwater."
    },
    "Plastic": {
        "material": "Thermoplastics (PET, HDPE, LDPE, PP, PS)",
        "disposal": "Rinse clean, squeeze/crush to conserve bin volume, and place in dry recyclables.",
        "reuse": "Repurpose clean containers for household organizing or workshop storage.",
        "tip": "Eliminating single-use plastics directly cuts municipal landfill accumulation and microplastic contamination."
    },
    "Paper": {
        "material": "Cellulose fibers",
        "disposal": "Keep clean and dry; place in designated dry paper recycling bins.",
        "reuse": "Use unprinted reverse sides for draft notes; shred clean paper for cushioning packages.",
        "tip": "Greasy, oily, or wet paper contaminates recycling batches and must be composted or sent to residual waste."
    },
    "Cardboard": {
        "material": "Corrugated paperboard",
        "disposal": "Remove plastic shipping tape, flatten boxes completely, and place in dry cardboard recycling.",
        "reuse": "Reuse for shipping, storage containers, or garden sheet mulching.",
        "tip": "Wet cardboard causes mechanical failures in paper pulping systems; always keep dry."
    },
    "Glass": {
        "material": "Silica container glass",
        "disposal": "Empty contents, rinse lightly, and place in glass recycling or bottle bank containers.",
        "reuse": "Wash and reuse glass jars for kitchen storage or dry pantry staples.",
        "tip": "Glass is 100% recyclable infinitely without degrading in quality or structural purity."
    },
    "Metal": {
        "material": "Aluminum or tin-plated steel",
        "disposal": "Empty and rinse clean; place in dry metal recycling.",
        "reuse": "Clean food tins can be repurposed as organizers, planters, or craft material.",
        "tip": "Recycling aluminum consumes 95% less energy than producing virgin metal from bauxite ore."
    },
    "Organic Waste": {
        "material": "Biodegradable organic matter",
        "disposal": "Deposit in wet-waste green bins or compost tumblers.",
        "reuse": "Process into nitrogen-rich compost or mulch to enrich home garden soil.",
        "tip": "Keeping organic waste out of anaerobic landfills prevents large-scale methane emissions."
    },
    "Textile": {
        "material": "Natural or synthetic fabrics (cotton, polyester, nylon)",
        "disposal": "Drop at textile banks, clothing drives, or municipal donation bins.",
        "reuse": "Mend worn garments, donate wearable pieces, or cut frayed cloth into cleaning wipes.",
        "tip": "Extending garment lifespans significantly cuts water consumption and chemical dyeing footprint."
    },
    "Hazardous Waste": {
        "material": "Corrosive, flammable, or toxic chemicals/batteries",
        "disposal": "Hand over exclusively to licensed municipal hazardous waste disposal depots.",
        "reuse": "Never reuse containers that held toxic chemicals or solvents.",
        "tip": "Never pour automotive fluids, paints, or battery acid down drains or household bins."
    },
    "General Waste": {
        "material": "Non-recyclable composite or soiled household refuse",
        "disposal": "Place in municipal residual/landfill waste bins.",
        "reuse": "Inspect whether individual components can be safely decoupled prior to disposal.",
        "tip": "Accurate pre-segregation prevents non-recyclable refuse from overloading sorting facilities."
    },
    "Other": {
        "material": "Inert construction materials, ceramics, masonry, or unclassified composites",
        "disposal": "Consult local municipal rubble and inert construction collection services.",
        "reuse": "Repurpose clean rubble for drainage beds, sub-base paths, or landscaping ballast.",
        "tip": "Segregate inert construction debris from domestic and organic streams before transport."
    }
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading EfficientNetV2B0 model weights (84.10% test accuracy benchmark)..."):
            r = requests.get(MODEL_URL, stream=True, timeout=300)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def build_gradcam_model():
    full_model = load_model()
    backbone_layer = next(l for l in full_model.layers if isinstance(l, tf.keras.Model))
    backbone = tf.keras.applications.EfficientNetV2B0(
        input_shape=(256, 256, 3), include_top=False, weights=None
    )
    backbone.set_weights(backbone_layer.get_weights())
    dense_layer = next(l for l in reversed(full_model.layers) if isinstance(l, tf.keras.layers.Dense))
    return backbone, dense_layer

def efficientnet_prediction(image: Image.Image):
    backbone, dense = build_gradcam_model()
    arr = np.asarray(image.resize((256, 256)), dtype=np.float32)
    x = tf.expand_dims(arr, 0)

    with tf.GradientTape() as tape:
        features = backbone(x, training=False)
        tape.watch(features)
        pooled = tf.reduce_mean(features, axis=[1, 2])
        logits = dense(pooled)
        probs = tf.nn.softmax(logits, axis=-1)[0]
        cid = tf.argmax(probs)
        score = logits[:, cid]

    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * features, axis=-1)
    cam = tf.maximum(cam[0], 0)
    cam /= (tf.reduce_max(cam) + 1e-8)
    cam_resized = tf.image.resize(cam[..., None], (256, 256)).numpy().squeeze()

    top3_indices = tf.argsort(probs, direction="DESCENDING")[:3].numpy()
    top3_results = [(classes[i], float(probs[i])) for i in top3_indices]

    return top3_results, cam_resized

def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def gemini_analysis(image: Image.Image, region_rule: str):
    client = get_gemini_client()
    if client is None:
        return None, "GEMINI_API_KEY not configured in st.secrets."

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()

    rag_context = json.dumps(KNOWLEDGE_BASE, indent=2)

    prompt = f"""
You are the PRIMARY open-world visual identifier for AI WasteWise.
Role:
1. Identify the physical item truthfully. You are NOT constrained to a 30-class list. Items like keyboards, phone chargers, laptops, masonry, concrete rubble, or car batteries must be recognized accurately.
2. Select EXACTLY ONE category from the allowed set.
3. Align your guidance with the selected regional rule and the RAG knowledge base.

Selected Regional Segregation Rule:
{region_rule}

Allowed Categories:
Electronic Waste, Plastic, Paper, Cardboard, Glass, Metal, Organic Waste, Textile, Hazardous Waste, General Waste, Other.

RAG Knowledge Base:
{rag_context}

Return strictly a JSON object:
{{
  "object_name": "<specific object or material name>",
  "category": "<one of the allowed categories>",
  "material": "<primary material>",
  "reason": "<short visual rationale for identification>",
  "disposal": "<disposal instructions adhering to regional rule>",
  "reuse": "<practical reuse or repair recommendation>",
  "sustainability_tip": "<short sustainability tip>"
}}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text.strip())
        if data.get("category") not in KNOWLEDGE_BASE:
            data["category"] = "Other"
        return data, None
    except Exception as e:
        return None, str(e)

# --- Sidebar: Configuration & Methodology ---
with st.sidebar:
    st.header("⚙️ System Configuration")
    selected_region = st.selectbox(
        "Municipal Segregation Standard:",
        list(REGIONAL_RULES.keys()),
        index=0
    )
    st.caption(REGIONAL_RULES[selected_region])
    
    cam_alpha = st.slider(
        "Grad-CAM Heatmap Opacity",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05
    )

    st.divider()
    st.header("📊 Technical Benchmarks")
    with st.expander("Custom Model Stats (EfficientNetV2B0)", expanded=False):
        st.markdown("""
        * **Input Resolution:** 256×256
        * **Classes:** 30 Household Waste Classes
        * **Test Accuracy:** **84.10%** (validated)
        * **Macro F1:** 83.93%
        * **Macro Precision:** 84.19%
        * **Macro Recall:** 84.09%
        * **Best Val Accuracy:** 84.44%
        * **Test Loss:** 0.4992
        """)
        st.caption("Baseline: MobileNetV3Large (79.70%), EfficientNetV2B0-224 (82.14%).")

    with st.expander("Cross-Dataset Generalization", expanded=False):
        st.markdown("""
        * **RealWaste (4,700 mapped images):**
          * Broad-category accuracy: 34.94%
          * Macro F1: 0.29 | Weighted F1: 0.32
        * **TACO (762 single-category images):**
          * Cross-dataset image-level accuracy: 12.34%
          * Macro F1: 0.10 | Weighted F1: 0.15
        """)
        st.caption("Demonstrates the closed-world bottleneck that motivated Gemini's integration as the primary open-world identifier.")

    with st.expander("Documented Confusion Pairs", expanded=False):
        st.markdown("""
        * `cardboard_packaging` vs `cardboard_boxes`
        * `aluminum_food_cans` vs `steel_food_cans`
        * `paper_cups` vs `styrofoam_cups`
        * `glass_beverage_bottles` vs `glass_food_jars`
        * `plastic_trash_bags` vs `plastic_shopping_bags`
        * `plastic_soda_bottles` vs `plastic_water_bottles`
        * `newspaper` vs `magazines`
        """)

# --- Main Page Layout ---
st.title("♻️ AI WasteWise")
st.write("Hybrid AI system combining **Gemini Vision (Primary Open-World Identifier)**, **EfficientNetV2B0 (Supporting 30-Class ML)**, **Grad-CAM Interpretability**, and **RAG Guidance**.")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    uploaded = st.file_uploader("📷 Upload a waste or household item image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Input", use_container_width=True)

with col_right:
    if uploaded and st.button("🔍 Execute Classification Pipeline", type="primary", use_container_width=True):
        with st.spinner("Analyzing with Gemini Vision & EfficientNetV2B0..."):
            top3, cam = efficientnet_prediction(image)
            gemini_res, gemini_err = gemini_analysis(image, REGIONAL_RULES[selected_region])

        top1_class, top1_prob = top3[0]
        ml_broad = broad.get(top1_class, "Other")

        if gemini_res:
            final_cat = gemini_res.get("category", "Other")
            final_name = gemini_res.get("object_name", "Identified Item")
            source = "Gemini Vision (Open-World Primary)"
        else:
            final_cat = ml_broad
            final_name = top1_class.replace("_", " ").title()
            source = "EfficientNetV2B0 (Supporting Fallback)"
            st.warning(f"Gemini fallback triggered: {gemini_err}")

        # Model Agreement & Out-of-Domain Diagnostic Badge
        st.subheader("🎯 Primary Classification Result")
        if gemini_res:
            if final_cat.lower() == ml_broad.lower():
                st.success(f"🟢 **High Model Agreement:** Both systems agreed on **{final_cat}**.")
            else:
                st.warning(
                    f"🟡 **Open-World / Taxonomy Discrepancy:** Gemini identified **{final_cat}** ({final_name}), "
                    f"while closed-set EfficientNet forced it to **{ml_broad}** (`{top1_class}`)."
                )

        st.markdown(f"### 🏷️ **{final_name}**")
        st.markdown(f"**Assigned Category:** `{final_cat}`  |  **Source:** *{source}*")

        # Tabbed Analysis Section
        tab_gemini, tab_ml, tab_rag = st.tabs([
            "✨ Gemini Vision (Primary)", 
            "🧠 EfficientNetV2B0 & Grad-CAM (Supporting)", 
            "🌱 RAG & Segregation Prep"
        ])

        with tab_gemini:
            if gemini_res:
                st.markdown(f"**Identified Item:** {gemini_res.get('object_name')}")
                st.markdown(f"**Assigned Category:** {gemini_res.get('category')}")
                st.markdown(f"**Detected Material:** {gemini_res.get('material')}")
                st.markdown(f"**Visual Justification:** {gemini_res.get('reason')}")
            else:
                st.info("Gemini Vision details unavailable for this session.")

        with tab_ml:
            st.markdown("#### Top-3 EfficientNetV2B0 Predictions (Closed 30-Class Set)")
            for item_name, prob in top3:
                clean_name = item_name.replace("_", " ").title()
                st.write(f"**{clean_name}** ({broad.get(item_name, 'Other')}): `{prob:.2%}`")
                st.progress(min(prob, 1.0))

            st.caption("Shows probability distribution across the closed 30-class taxonomy.")

            st.markdown("#### 🔥 Grad-CAM Attention Map")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(image.resize((256, 256)))
            ax.imshow(cam, cmap="jet", alpha=cam_alpha)
            ax.axis("off")
            ax.set_title(f"Focus on '{top1_class}' (Attention Map)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Grad-CAM explains EfficientNetV2B0 activations only. It does not explain Gemini.")

        with tab_rag:
            rag_info = KNOWLEDGE_BASE.get(final_cat, KNOWLEDGE_BASE["Other"])
            disposal_text = gemini_res.get("disposal", rag_info["disposal"]) if gemini_res else rag_info["disposal"]
            reuse_text = gemini_res.get("reuse", rag_info["reuse"]) if gemini_res else rag_info["reuse"]
            tip_text = gemini_res.get("sustainability_tip", rag_info["tip"]) if gemini_res else rag_info["tip"]
            material_text = gemini_res.get("material", rag_info["material"]) if gemini_res else rag_info["material"]

            st.markdown(f"**Region Standard:** `{selected_region}`")
            st.markdown(f"**Primary Material:** {material_text}")
            st.markdown(f"**🗑️ Disposal Route:** {disposal_text}")
            st.markdown(f"**🔄 Reuse / Repair:** {reuse_text}")
            st.markdown(f"**🌍 Sustainability Impact:** {tip_text}")

            st.markdown("#### ✅ Recycling Prep Checklist")
            st.checkbox("Rinsed and free of organic/food residues", value=False)
            st.checkbox("Completely dry before binning (prevents paper batch contamination)", value=False)
            st.checkbox("Separated composite parts (e.g., plastic film, bottle caps, tape)", value=False)
            st.checkbox("Flattened / compacted to minimize collection container volume", value=False)

    elif not uploaded:
        st.info("👆 Upload an image in the left panel to execute the hybrid pipeline.")
