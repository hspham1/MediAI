"""
streamlit as frontend application for mediAI
"""
import streamlit as st
import requests
import base64
import io
from PIL import Image

def encode_image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

def decode_base64_to_image(img_b64: str) -> Image.Image:
    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return image

def generate_mri_button(uploaded_file, base_url, img_width):
    # create a button to submit the image to the backend
    if st.button("Generate MRI Scan"):
        if uploaded_file is not None:
            
            col1, col2 = st.columns(2) # two columns to display input and output images side by side

            with col1:
                st.header("Input CT")
                st.image(uploaded_file, caption="Uploaded CT Scan", width=img_width)
            
            # Read the uploaded image
            ct_image = Image.open(uploaded_file).convert("RGB")
            
            # Encode image to base64
            ct_image_b64 = encode_image_to_base64(ct_image)
            
            # Prepare payload
            payload = {"ct_image": ct_image_b64}
            
            # Send POST request to backend API
            try:
                response = requests.post(f"{base_url}/generate_mri", json=payload, timeout=60)
            except Exception as e:
                st.error(f"Request failed: {e}")
                return
            
            st.write(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    mri_image_b64 = data.get("mri_image")
                    if not mri_image_b64:
                        st.error("No 'mri_image' in response")
                        st.json(data)
                        return
                    img_bytes = base64.b64decode(mri_image_b64)
                    mri_image = decode_base64_to_image(mri_image_b64)

                    with col2:
                        st.header("Generated MRI")
                        st.image(mri_image, caption="Generated MRI Scan", width=img_width)

                    # Provide download button for the generated MRI image
                    st.download_button(
                        label="Download MRI Image",
                        data=img_bytes,
                        file_name="generated_mri.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Error decoding response: {e}")
                    st.text(response.text)
            else:
                st.error(f"Error in generating MRI scan (status {response.status_code}). Please try again.")
                st.text(response.text)
        else:
            st.warning("Please upload a CT scan image first.")

def main():
    st.title("MediAI: CT to MRI Translation")
    st.write("Upload a CT scan image to generate the corresponding MRI scan using our AI model.")

    img_width = st.slider("Display width (px)", min_value=200, max_value=800, value=400)


    uploaded_file = st.file_uploader("Choose a CT scan image...", type=["png", "jpg", "jpeg"])
    base_url = "http://localhost:8000"  # backend API URL

    generate_mri_button(uploaded_file, base_url, img_width)
    

if __name__ == "__main__":
    main()