import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
@st.cache_resource
def load_blip_resources():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

def main():
    st.title("Muhammad Ikrma Sharaf")
    st.title("Image Captioning")

    processor, model = load_blip_resources()

    # Upload image
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:

        st.image(uploaded_image, caption="Uploaded Image", width=500)

        # Perform image captioning on button click
        if st.button("Generate Caption"):
            # Display loading spinner
            with st.spinner("Generating caption..."):
                # Perform image captioning
                text = "a photography of"
                raw_image = Image.open(uploaded_image).convert('RGB')
                inputs = processor(raw_image, text, return_tensors="pt")

                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)

                # Display generated caption
                st.write("Generated Caption:", caption)

if __name__ == "__main__":
    main()
