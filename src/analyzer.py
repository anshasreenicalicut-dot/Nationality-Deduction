
import cv2
import numpy as np
from deepface import DeepFace

from .dress_color import get_dress_color


# These are model demographic labels, not nationalities.
GROUP_MAPPING = {
    "indian": "Indian",
    "black": "African/Black demographic group",
    "white": "White demographic group",
    "asian": "Other",
    "middle eastern": "Other",
    "latino hispanic": "Other",
}


def normalize_group(race):
    label = str(race).lower().strip()
    return GROUP_MAPPING.get(label, "Other")


def analyze_image(image):
    """
    Analyze an RGB image using DeepFace.

    Returns:
        group
        detected_race
        age
        emotion
        dress_color
        group_confidence
        emotion_confidence
    """

    if image is None:
        raise ValueError("No image was provided.")

    image = np.asarray(image)

    if image.size == 0:
        raise ValueError("The uploaded image is empty.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Please upload a valid colour image.")

    # Convert RGB to BGR.
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # -----------------------------------------------------
    # FACE, AGE, EMOTION AND DEMOGRAPHIC ANALYSIS
    # -----------------------------------------------------

    try:
        results = DeepFace.analyze(
            img_path=image_bgr,
            actions=["age", "emotion", "race"],
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            silent=True,
        )

    except Exception as exc:
        message = str(exc)

        if "haarcascade_frontalface_default.xml" in message:
            raise RuntimeError(
                "OpenCV cannot find its Haar Cascade XML file. "
                "Follow the OpenCV repair commands provided below."
            ) from exc

        raise RuntimeError(
            "DeepFace could not analyze this image. "
            "Try a clear, front-facing image with good lighting. "
            f"Technical details: {message}"
        ) from exc

    if isinstance(results, list):
        if not results:
            raise RuntimeError(
                "No face was detected. Try another clear image."
            )
        result = results[0]
    elif isinstance(results, dict):
        result = results
    else:
        raise RuntimeError("DeepFace returned an unexpected result.")

    # -----------------------------------------------------
    # AGE
    # -----------------------------------------------------

    try:
        age = int(round(float(result.get("age", "Unknown"))))
    except (TypeError, ValueError, OverflowError):
        age = "Unknown"

    # -----------------------------------------------------
    # EMOTION
    # -----------------------------------------------------

    emotion = str(result.get("dominant_emotion", "Unknown"))
    emotion_scores = result.get("emotion", {})
    emotion_confidence = None

    if isinstance(emotion_scores, dict):
        score = emotion_scores.get(emotion)
        if score is not None:
            try:
                emotion_confidence = float(score)
            except (TypeError, ValueError):
                pass

    # -----------------------------------------------------
    # DEMOGRAPHIC CATEGORY
    # -----------------------------------------------------

    race = str(result.get("dominant_race", "Unknown"))
    group = normalize_group(race)
    race_scores = result.get("race", {})
    group_confidence = None

    if isinstance(race_scores, dict):
        score = race_scores.get(race)
        if score is not None:
            try:
                group_confidence = float(score)
            except (TypeError, ValueError):
                pass

    # -----------------------------------------------------
    # APPROXIMATE CLOTHING COLOUR
    # -----------------------------------------------------

    dress_color = get_dress_color(image_bgr)

    return {
        "group": group,
        "detected_race": race,
        "age": age,
        "emotion": emotion,
        "dress_color": dress_color,
        "group_confidence": group_confidence,
        "emotion_confidence": emotion_confidence,
    }
