from deepface import DeepFace
import cv2
import numpy as np

from src.dress_color import get_dress_color


# Map DeepFace race labels to the categories
# required by the academic project.
#
# IMPORTANT:
# These are demographic/race groups, NOT verified nationality.
GROUP_MAPPING = {
    "indian": "Indian",
    "black": "African",
    "white": "US/White",
    "asian": "Other",
    "middle eastern": "Other",
    "latino hispanic": "Other",
}


def normalize_group(race):
    race = str(race).lower().strip()

    return GROUP_MAPPING.get(race, "Other")


def analyze_image(image):
    """
    Analyze one image.

    Returns:
        dict containing group, age, emotion,
        dress colour and confidence information.
    """

    if image is None:
        raise ValueError("Image is empty.")

    # Convert RGB -> BGR for OpenCV / DeepFace
    if len(image.shape) == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    try:
        results = DeepFace.analyze(
            img_path=bgr,
            actions=["age", "emotion", "race"],
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            silent=True
        )

    except Exception as e:
        raise RuntimeError(
            f"Face analysis failed: {str(e)}"
        )

    # DeepFace may return list for multiple faces
    if isinstance(results, list):
        if len(results) == 0:
            raise RuntimeError("No face detected.")

        result = results[0]
    else:
        result = results

    # -------------------------
    # AGE
    # -------------------------

    age = result.get("age", "Unknown")

    try:
        age = int(round(float(age)))
    except Exception:
        age = "Unknown"

    # -------------------------
    # EMOTION
    # -------------------------

    emotion = result.get("dominant_emotion", "Unknown")

    emotion_scores = result.get("emotion", {})

    emotion_confidence = None

    if isinstance(emotion_scores, dict):
        if emotion in emotion_scores:
            try:
                emotion_confidence = float(
                    emotion_scores[emotion]
                )
            except Exception:
                emotion_confidence = None

    # -------------------------
    # RACE / DEMOGRAPHIC GROUP
    # -------------------------

    race = result.get(
        "dominant_race",
        "Unknown"
    )

    group = normalize_group(race)

    race_scores = result.get("race", {})

    group_confidence = None

    if isinstance(race_scores, dict):
        if race in race_scores:
            try:
                group_confidence = float(
                    race_scores[race]
                )
            except Exception:
                group_confidence = None

    # -------------------------
    # DRESS COLOUR
    # -------------------------

    dress_color = get_dress_color(bgr)

    # -------------------------
    # CONDITIONAL OUTPUT
    # -------------------------

    output = {
        "group": group,
        "detected_race": race,
        "age": age,
        "emotion": emotion,
        "dress_color": dress_color,
        "group_confidence": group_confidence,
        "emotion_confidence": emotion_confidence
    }

    return output