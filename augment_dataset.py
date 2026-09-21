from pathlib import Path

import cv2
import numpy as np
import albumentations as A


ROOT = Path(__file__).resolve().parent

IMAGE_DIR = (
    ROOT
    / "dataset"
    / "images"
    / "train"
)

LABEL_DIR = (
    ROOT
    / "dataset"
    / "labels"
    / "train"
)

AUG_IMAGE_DIR = (
    ROOT
    / "dataset"
    / "images"
    / "train_augmented"
)

AUG_LABEL_DIR = (
    ROOT
    / "dataset"
    / "labels"
    / "train_augmented"
)


AUG_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUG_LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


transform = A.Compose(

    [

        A.HorizontalFlip(
            p=0.5
        ),

        A.Rotate(
            limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.GaussNoise(
            std_range=(0.02, 0.08),
            p=0.3
        ),

        A.OneOf(

            [

                A.RandomGamma(
                    gamma_limit=(80, 120)
                ),

                A.CLAHE(
                    clip_limit=2.0
                ),

            ],

            p=0.3

        ),

    ],

    bbox_params=A.BboxParams(

        format="yolo",

        label_fields=[
            "class_labels"
        ],

        min_visibility=0.2

    )

)


def read_labels(label_file):

    boxes = []

    classes = []

    if not label_file.exists():

        return boxes, classes

    lines = label_file.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        box = [
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4])
        ]

        classes.append(class_id)

        boxes.append(box)

    return boxes, classes


def save_labels(
    label_file,
    boxes,
    classes
):

    lines = []

    for class_id, box in zip(
        classes,
        boxes
    ):

        x, y, w, h = box

        lines.append(

            f"{class_id} "
            f"{x:.6f} "
            f"{y:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"

        )

    label_file.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def augment_dataset():

    image_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]:

        image_files.extend(
            IMAGE_DIR.glob(extension)
        )

    print(
        f"Found {len(image_files)} "
        f"training images."
    )

    generated = 0

    for image_file in image_files:

        label_file = (
            LABEL_DIR
            / f"{image_file.stem}.txt"
        )

        image = cv2.imread(
            str(image_file)
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        boxes, classes = read_labels(
            label_file
        )

        if not boxes:
            continue

        augmented = transform(

            image=image,

            bboxes=boxes,

            class_labels=classes

        )

        output_image = cv2.cvtColor(

            augmented["image"],

            cv2.COLOR_RGB2BGR

        )

        output_name = (
            f"{image_file.stem}_aug.jpg"
        )

        output_image_path = (
            AUG_IMAGE_DIR
            / output_name
        )

        output_label_path = (
            AUG_LABEL_DIR
            / f"{image_file.stem}_aug.txt"
        )

        cv2.imwrite(

            str(output_image_path),

            output_image

        )

        save_labels(

            output_label_path,

            augmented["bboxes"],

            augmented["class_labels"]

        )

        generated += 1

    print(
        f"Generated {generated} "
        f"augmented images."
    )


if __name__ == "__main__":

    augment_dataset()