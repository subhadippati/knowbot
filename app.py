import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from vit_model import VisionTransformer

# ---------- CONFIG ---------- #
st.set_page_config(page_title="KnowBot - ViT Image Classifier", page_icon="🤖", layout="centered")

# ---------- TITLE SECTION ---------- #
st.markdown("""
    <h1 style='text-align: center; color: #6C63FF;'>🤖 KnowBot</h1>
    <h4 style='text-align: center; color: grey;'>Upload an image and let the Vision Transformer tell you what it is!</h4>
    <hr style='margin-top: 10px; margin-bottom: 25px;'>
""", unsafe_allow_html=True)

# ---------- CLASS LABELS ---------- #
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ---------- LOAD MODEL ---------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(img_size=32, patch_size=4, in_channels=3, n_classes=10,
                          embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1)
model.load_state_dict(torch.load('vit_model.pth', map_location=device))
model.to(device)
model.eval()

# ---------- IMAGE TRANSFORM ---------- #
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------- FILE UPLOADER ---------- #
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.markdown("### 🖼️ Preview")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # ---------- PREDICTION ---------- #
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    # ---------- RESULT DISPLAY ---------- #
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align:center;">
            <h2 style="color:#00C851;">✅ Prediction: {predicted_class.upper()}</h2>
            <h4 style="color:#33b5e5;">Confidence: {confidence_score:.2f}%</h4>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align:center; color: #888;'>Awaiting image upload...</p>", unsafe_allow_html=True)
