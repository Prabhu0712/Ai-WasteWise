
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "/content/drive/MyDrive/AI Wastewise/efficientnetv2b0_256.keras"

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

knowledge = {
"steel_food_cans":("Steel/metal","Empty and rinse, then place in recyclable metal waste.","Reuse for storage or planters.","Rinsing removes residue and improves recyclability."),
"aluminum_food_cans":("Aluminum","Rinse and place with recyclable metal waste.","Reuse for crafts or storage.","Aluminum can be recycled repeatedly."),
"aluminum_soda_cans":("Aluminum","Empty, rinse and place in recyclable metal waste.","Reuse for crafts.","Avoid putting aluminum cans in general waste."),
"plastic_water_bottles":("PET plastic","Empty and place in appropriate plastic recycling.","Reuse temporarily for non-food purposes.","Recycling is preferable to single-use disposal."),
"plastic_soda_bottles":("PET plastic","Empty and place in plastic recycling.","Reuse for storage or crafts.","Keep bottles separate from contaminated waste."),
"plastic_food_containers":("Plastic","Empty and clean before plastic recycling.","Reuse for storage when appropriate.","Heavily contaminated plastic may not be recyclable."),
"cardboard_boxes":("Cardboard","Flatten and place with dry paper/cardboard recycling.","Reuse for packaging and storage.","Keep cardboard dry and clean."),
"cardboard_packaging":("Paper/cardboard","Place clean, dry packaging with paper/cardboard recycling.","Reuse as packaging material.","Remove food contamination before recycling."),
"glass_beverage_bottles":("Glass","Place in appropriate glass recycling.","Reuse as containers or decorative items.","Handle broken glass carefully."),
"glass_food_jars":("Glass","Empty and rinse before glass recycling.","Reuse for household storage.","Separate lids if required locally."),
"paper_cups":("Paper with possible plastic coating","Check local rules for coated cups.","Generally single-use.","Do not assume all paper cups are recyclable."),
"plastic_trash_bags":("Plastic film","Use plastic-film collection where available.","Reuse for waste collection.","Keep recyclable materials separate."),
"plastic_straws":("Plastic","Follow local plastic-waste guidelines.","Generally difficult to reuse safely.","Consider reusable alternatives."),
"shoes":("Mixed materials","Donate or use footwear/textile collection where available.","Donate or repair if usable.","Avoid sending usable footwear directly to landfill."),
"food_waste":("Organic matter","Place in organic/compost waste where available.","Compost suitable food scraps.","Keep organic waste separate from recyclables."),
"coffee_grounds":("Organic matter","Add to compost or organic waste where permitted.","Compost or use as soil amendment.","Keep separate from dry recyclables."),
"eggshells":("Organic/mineral material","Place in compost/organic waste where supported.","Crush and add to compost.","Clean shells before reuse."),
"tea_bags":("Organic material plus possible synthetic components","Check bag material before composting.","Compost tea leaves when suitable.","Some tea bags contain plastic fibers."),
"disposable_plastic_cutlery":("Plastic","Follow local plastic-waste guidelines.","Limited reuse only when appropriate.","Reusable cutlery reduces plastic waste."),
"styrofoam_cups":("Expanded polystyrene","Check whether local facilities accept polystyrene.","Generally single-use.","Check local collection rules."),
"styrofoam_food_containers":("Expanded polystyrene","Check local recycling/disposal facilities.","Generally single-use.","Reduce use where reusable containers are available.")
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def build_gradcam():
    trained = load_model()
    old_backbone = next(l for l in trained.layers if isinstance(l, tf.keras.Model))
    backbone = tf.keras.applications.EfficientNetV2B0(
        input_shape=(256,256,3), include_top=False, weights=None)
    backbone.set_weights(old_backbone.get_weights())
    dense = next(l for l in trained.layers[::-1] if isinstance(l, tf.keras.layers.Dense))
    return backbone, dense

def predict_and_cam(img):
    backbone, dense = build_gradcam()
    x = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)
    with tf.GradientTape() as tape:
        features = backbone(x, training=False)
        tape.watch(features)
        pooled = tf.reduce_mean(features, axis=[1,2])
        logits = dense(pooled)
        cid = tf.argmax(logits[0])
        score = logits[:, cid]
    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1,2))
    cam = tf.reduce_sum(weights[:,None,None,:] * features, axis=-1)
    cam = tf.maximum(cam[0],0)
    cam /= tf.reduce_max(cam)+1e-8
    cam = tf.image.resize(cam[...,None],(256,256)).numpy().squeeze()
    probs = tf.nn.softmax(logits)[0].numpy()
    return int(cid), float(probs[int(cid)]), cam

st.set_page_config(page_title="AI WasteWise", page_icon="♻️", layout="centered")
st.title("♻️ AI WasteWise")
st.caption("AI-powered waste classification, explainability and sustainable disposal guidance.")

uploaded = st.file_uploader("Upload a waste image", type=["jpg","jpeg","png"])

if uploaded:
    img = tf.keras.utils.load_img(uploaded, target_size=(256,256))
    st.image(img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing waste..."):
        cid, conf, cam = predict_and_cam(img)

    label = classes[cid]
    info = knowledge.get(label, ("Unknown","Check local waste-management guidelines.","Consider safe reuse where appropriate.","Follow local disposal rules."))

    st.success(f"Prediction: {label.replace('_',' ').title()}")
    st.metric("Confidence", f"{conf:.2%}")

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img)
    ax.imshow(cam, cmap="jet", alpha=0.45)
    ax.axis("off")
    ax.set_title(f"Grad-CAM — {label.replace('_',' ').title()}")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("♻️ Sustainability Guidance")
    st.write(f"**Material:** {info[0]}")
    st.write(f"**Disposal:** {info[1]}")
    st.write(f"**Reuse:** {info[2]}")
    st.write(f"**Sustainability Tip:** {info[3]}")

    st.info(
        "AI-generated guidance is informational. Local waste-management rules "
        "should be followed for final disposal decisions."
    )
else:
    st.write("Upload an image to start the AI WasteWise analysis.")
