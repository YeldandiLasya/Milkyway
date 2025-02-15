import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import torch
from diffusers import StableDiffusionPipeline


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load Stable Diffusion model
@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda" if torch.cuda.is_available() else "cpu")
    return model

model = load_model()

# Streamlit UI
st.title("Hand Gesture-Controlled Generative AI 🎨🖐️")
st.text("Show different gestures to control AI-generated images!")

# Capture video
cap = cv2.VideoCapture(0)

if st.button("Start Camera"):
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror the image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand detection
        results = hands.process(rgb_frame)
        gesture = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark coordinates
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Simple Gesture Detection: Thumb-Up or Index-Up
                if thumb_tip.y < index_tip.y:
                    gesture = "Thumb Up"
                else:
                    gesture = "Index Up"

        # Display gesture text
        cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show video feed
        st.image(frame, channels="BGR", use_column_width=True)

        # AI Image Generation based on gesture
        if gesture == "Thumb Up":
            st.write("Generating a futuristic city 🌆...")
            image = model("futuristic city").images[0]
            st.image(image, caption="Generated Image")

        elif gesture == "Index Up":
            st.write("Generating a beautiful landscape 🌄...")
            image = model("beautiful landscape").images[0]
            st.image(image, caption="Generated Image")

    cap.release()
    cv2.destroyAllWindows()
