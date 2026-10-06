import streamlit as st
from PIL import Image
import numpy as np

from src.analyzer import analyze_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Face Attribute AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }

    .result-title {
        font-size: 20px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Face Attribute AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Demographic Group • Age • Emotion • Dress Colour
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Project Information")

    st.write(
        """
        This application analyzes an uploaded face image
        and displays:

        • Demographic group  
        • Estimated age  
        • Facial emotion  
        • Approximate dress colour
        """
    )

    st.divider()

    st.info(
        """
        The demographic output is based on facial
        demographic/race categories provided by the
        underlying model. It should not be interpreted
        as verified nationality.
        """
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a face image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        image_array = np.array(image)

        # ----------------------------------------------------
        # PREVIEW
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📷 Input Image")

            st.image(
                image,
                caption="Uploaded Image",
                width="stretch"
            )

        with col2:

            st.subheader("🔍 Analysis")

            analyze_button = st.button(
                "🚀 Analyze Image",
                type="primary",
                use_container_width=True
            )

        # ----------------------------------------------------
        # ANALYSIS
        # ----------------------------------------------------

        if analyze_button:

            with st.spinner(
                "Analyzing face... Please wait."
            ):

                try:

                    result = analyze_image(
                        image_array
                    )

                except Exception as e:

                    st.error(
                        f"❌ Analysis failed: {e}"
                    )

                    st.stop()

            st.divider()

            # ------------------------------------------------
            # RESULT HEADER
            # ------------------------------------------------

            st.header("📊 Prediction Results")

            # ------------------------------------------------
            # BASIC RESULT CARDS
            # ------------------------------------------------

            r1, r2, r3, r4 = st.columns(4)

            with r1:

                st.metric(
                    "Demographic Group",
                    result["group"]
                )

            with r2:

                st.metric(
                    "Estimated Age",
                    str(result["age"])
                )

            with r3:

                st.metric(
                    "Emotion",
                    str(result["emotion"]).title()
                )

            with r4:

                st.metric(
                    "Dress Colour",
                    result["dress_color"]
                )

            # ------------------------------------------------
            # CONDITIONAL LOGIC
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📌 Assignment Output"
            )

            group = result["group"]

            if group == "Indian":

                st.success(
                    "Indian group detected"
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.write("### 👤 Age")
                    st.write(result["age"])

                with c2:
                    st.write("### 😊 Emotion")
                    st.write(
                        str(
                            result["emotion"]
                        ).title()
                    )

                with c3:
                    st.write("### 👗 Dress Colour")
                    st.write(
                        result["dress_color"]
                    )

            elif group == "US/White":

                st.info(
                    "US/White demographic group"
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.write("### 👤 Age")
                    st.write(result["age"])

                with c2:
                    st.write("### 😊 Emotion")
                    st.write(
                        str(
                            result["emotion"]
                        ).title()
                    )

            elif group == "African":

                st.info(
                    "African/Black demographic group"
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.write("### 😊 Emotion")
                    st.write(
                        str(
                            result["emotion"]
                        ).title()
                    )

                with c2:
                    st.write("### 👗 Dress Colour")
                    st.write(
                        result["dress_color"]
                    )

            else:

                st.warning(
                    "Other demographic group"
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.write("### 🌍 Group")
                    st.write(
                        result["group"]
                    )

                with c2:
                    st.write("### 😊 Emotion")
                    st.write(
                        str(
                            result["emotion"]
                        ).title()
                    )

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📈 Model Information"
            )

            c1, c2 = st.columns(2)

            with c1:

                if result["group_confidence"] is not None:

                    st.write(
                        "Demographic prediction confidence"
                    )

                    st.progress(
                        min(
                            max(
                                result[
                                    "group_confidence"
                                ] / 100,
                                0
                            ),
                            1
                        )
                    )

                    st.write(
                        f"{result['group_confidence']:.2f}%"
                    )

            with c2:

                if result["emotion_confidence"] is not None:

                    st.write(
                        "Emotion prediction confidence"
                    )

                    st.progress(
                        min(
                            max(
                                result[
                                    "emotion_confidence"
                                ] / 100,
                                0
                            ),
                            1
                        )
                    )

                    st.write(
                        f"{result['emotion_confidence']:.2f}%"
                    )


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.info(
        "👆 Upload an image above to start analysis."
    )

    st.markdown(
        """
        ### 🔄 System Workflow

        **Image Upload**
        ↓

        **Face Detection**
        ↓

        **Demographic Group Analysis**
        ↓

        **Age Prediction**
        ↓

        **Emotion Prediction**
        ↓

        **Dress Colour Estimation**
        ↓

        **Conditional Result Display**
        """
    )