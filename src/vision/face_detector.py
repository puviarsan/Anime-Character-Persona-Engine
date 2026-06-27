"""
==========================================================
Project : Anime Character Persona Engine

Module  : face_detector.py

Phase   : Phase 3

Description
-----------
YOLOv8 Anime Face Detection Pipeline

1. Scan Dataset
2. Load Images
3. Detect Faces
4. Crop Faces
5. Save Faces
6. Generate Report
==========================================================
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import pandas as pd

from ultralytics import YOLO

from config.paths import (
    RAW_IMAGES_DIR,
    FACE_OUTPUT_DIR,
    FACE_REPORT_FILE,
    YOLO_FACE_MODEL
)

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# Face Detector
# ==========================================================

class FaceDetector:

    """
    YOLOv8 Face Detection Pipeline
    """

    def __init__(self):

        self.dataset = []

        self.results = []

        self.image = None

        self.current_image = None

        self.current_character = None

        self.start_time = 0

        self.end_time = 0

        logger.info("Loading YOLOv8 Face Model...")

        self.detector = YOLO(
            str(YOLO_FACE_MODEL)
        )

        logger.info("YOLOv8 Model Loaded Successfully.")
        # ======================================================
    # Scan Dataset
    # ======================================================

    def scan_dataset(self):

        logger.info(
            "Scanning image dataset..."
        )

        self.dataset = []

        image_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        for character_folder in RAW_IMAGES_DIR.iterdir():

            if not character_folder.is_dir():
                continue

            character = character_folder.name

            for image_path in character_folder.iterdir():

                if image_path.suffix.lower() not in image_extensions:
                    continue

                self.dataset.append(
                    {
                        "character": character,
                        "image_path": image_path
                    }
                )

        logger.info(
            f"Images Found : {len(self.dataset)}"
        )

        return self.dataset
        # ======================================================
    # Validate Dataset
    # ======================================================

    def validate_dataset(self):

        if len(self.dataset) == 0:

            raise ValueError(
                "No images found in data/raw/images."
            )

        logger.info(
            "Dataset validation successful."
        )

        # ======================================================
    # Load Image
    # ======================================================

    def load_image(self, image_info):

        self.current_character = image_info["character"]

        self.current_image = image_info["image_path"]

        logger.info(
            f"Loading : {self.current_image.name}"
        )

        self.image = cv2.imread(
            str(self.current_image)
        )

        if self.image is None:

            raise ValueError(
                f"Unable to load {self.current_image}"
            )

        return self.image
        # ======================================================
    # Image Information
    # ======================================================

    def image_information(self):

        height, width, channels = self.image.shape

        return {

            "character": self.current_character,

            "filename": self.current_image.name,

            "width": width,

            "height": height,

            "channels": channels

        }
        # ======================================================
    # Detect Faces
    # ======================================================

    def detect_faces(self):

        predictions = self.detector.predict(

            source=self.image,

            conf=0.60,

            verbose=False

        )

        faces = []

        result = predictions[0]

        if len(result.boxes) == 0:

            logger.info(

                f"{self.current_image.name} -> 0 face(s)"

            )

            return faces

        for index, box in enumerate(result.boxes):

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            confidence = float(

                box.conf[0]

            )

            x = int(x1)

            y = int(y1)

            w = int(x2 - x1)

            h = int(y2 - y1)

            faces.append(

                {

                    "face_id": index + 1,

                    "character": self.current_character,

                    "filename": self.current_image.name,

                    "x": x,

                    "y": y,

                    "width": w,

                    "height": h,

                    "area": w * h,

                    "center_x": x + w // 2,

                    "center_y": y + h // 2,

                    "confidence": round(

                        confidence,

                        3

                    )

                }

            )

        # ==================================================
        # Keep only the largest detected face
        # ==================================================

        if len(faces) > 1:

            largest_face = max(

                faces,

                key=lambda face: face["area"]

            )

            faces = [largest_face]

            faces[0]["face_id"] = 1

        logger.info(

            f"{self.current_image.name} -> {len(faces)} face(s)"

        )

        return faces
        # ======================================================
    # Draw Bounding Boxes
    # ======================================================

    def draw_faces(

        self,

        faces

    ):

        image = self.image.copy()

        for face in faces:

            x = face["x"]

            y = face["y"]

            w = face["width"]

            h = face["height"]

            confidence = face["confidence"]

            cv2.rectangle(

                image,

                (x, y),

                (x + w, y + h),

                (0, 255, 0),

                2

            )

            cv2.putText(

                image,

                f"{confidence:.2f}",

                (x, y - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 0),

                2

            )

        return image
        # ======================================================
    # Face Statistics
    # ======================================================

    def face_statistics(

        self,

        faces

    ):

        if len(faces) == 0:

            return

        dataframe = pd.DataFrame(

            faces

        )

        print()

        print("=" * 70)

        print(

            f"FACE STATISTICS : {self.current_image.name}"

        )

        print("=" * 70)

        print(dataframe)

        print("=" * 70)

        print()
    
        # ======================================================
    # Process One Image
    # ======================================================

    def process_image(

        self,

        image_info

    ):

        self.load_image(

            image_info

        )

        info = self.image_information()

        faces = self.detect_faces()

        self.face_statistics(

            faces

        )

        return info, faces
        # ======================================================
    # Process Dataset
    # ======================================================

    def process_dataset(self):

        logger.info(

            "Processing dataset..."

        )

        self.results = []

        self.start_time = time.time()

        for image_info in self.dataset:

            info, faces = self.process_image(

                image_info

            )

            for face in faces:

                row = {}

                row.update(info)

                row.update(face)

                self.results.append(

                    row

                )

        logger.info(

            f"Detected {len(self.results)} total face(s)."

        )

        return self.results
        # ======================================================
    # Crop and Save Faces
    # ======================================================

    def crop_and_save_faces(self):

        logger.info(
            "Cropping and saving faces..."
        )

        for result in self.results:

            character = result["character"]

            output_dir = FACE_OUTPUT_DIR / character

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            image_path = (
                RAW_IMAGES_DIR /
                character /
                result["filename"]
            )

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            x = result["x"]
            y = result["y"]
            w = result["width"]
            h = result["height"]

            face = image[
                y:y+h,
                x:x+w
            ]

            output_file = (
                output_dir /
                f"{Path(result['filename']).stem}_face_{result['face_id']}.jpg"
            )

            cv2.imwrite(
                str(output_file),
                face
            )

            result["cropped_face"] = str(output_file)

        logger.info(
            "Face cropping completed."
        )

        # ======================================================
    # Save Detection Images
    # ======================================================

    def save_detection_images(self):

        logger.info(
            "Saving detection images..."
        )

        detection_dir = (
            FACE_OUTPUT_DIR /
            "detections"
        )

        detection_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        grouped = {}

        for row in self.results:

            key = (
                row["character"],
                row["filename"]
            )

            grouped.setdefault(
                key,
                []
            ).append(row)

        for (character, filename), faces in grouped.items():

            image = cv2.imread(
                str(
                    RAW_IMAGES_DIR /
                    character /
                    filename
                )
            )

            if image is None:
                continue

            for face in faces:

                x = face["x"]
                y = face["y"]
                w = face["width"]
                h = face["height"]
                confidence = face["confidence"]

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"{confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            output_file = (
                detection_dir /
                f"{Path(filename).stem}_detected.jpg"
            )

            cv2.imwrite(
                str(output_file),
                image
            )

        logger.info(
            "Detection images saved."
        )

        # ======================================================
    # Save Report
    # ======================================================

    def save_report(self):

        logger.info(
            "Saving face detection report..."
        )

        dataframe = pd.DataFrame(
            self.results
        )

        dataframe.to_csv(
            FACE_REPORT_FILE,
            index=False
        )

        logger.info(
            f"Saved:\n{FACE_REPORT_FILE}"
        )

        return dataframe
        # ======================================================
    # Dataset Statistics
    # ======================================================

    def dataset_statistics(self):

        print()

        print("=" * 70)
        print("FACE DETECTION DATASET")
        print("=" * 70)

        print(f"Images Processed : {len(self.dataset)}")
        print(f"Characters       : {len(set(item['character'] for item in self.dataset))}")
        print(f"Faces Detected   : {len(self.results)}")

        if len(self.results) > 0:

            dataframe = pd.DataFrame(
                self.results
            )

            print(f"Average Confidence : {round(dataframe['confidence'].mean(),3)}")

        else:

            print("Average Confidence : N/A")

        print("=" * 70)
        print()

        # ======================================================
    # Performance Report
    # ======================================================

    def performance_report(self):

        self.end_time = time.time()

        elapsed = round(
            self.end_time -
            self.start_time,
            3
        )

        print()

        print("=" * 70)
        print("PIPELINE PERFORMANCE")
        print("=" * 70)
        print(f"Execution Time : {elapsed} sec")
        print("=" * 70)
        print()

        # ======================================================
    # Summary
    # ======================================================

    def summary(self):

        print()

        print("=" * 70)
        print("YOLOv8 FACE DETECTION SUMMARY")
        print("=" * 70)

        print(f"Total Images      : {len(self.dataset)}")
        print(f"Total Characters  : {len(set(item['character'] for item in self.dataset))}")
        print(f"Total Faces       : {len(self.results)}")

        if len(self.results) > 0:

            dataframe = pd.DataFrame(self.results)

            print(f"Largest Face Area : {dataframe['area'].max()}")
            print(f"Average Face Area : {round(dataframe['area'].mean(), 2)}")
            print(f"Average Confidence: {round(dataframe['confidence'].mean(), 3)}")

        else:

            print("Largest Face Area : N/A")
            print("Average Face Area : N/A")
            print("Average Confidence: N/A")

        print("=" * 70)
        print()

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("Anime Character Persona Engine")
    logger.info("Phase 3 - YOLOv8 Face Detector")
    logger.info("=" * 70)

    detector = FaceDetector()

    try:

        detector.scan_dataset()

        detector.validate_dataset()

        detector.process_dataset()

        detector.crop_and_save_faces()

        detector.save_detection_images()

        detector.save_report()

        detector.dataset_statistics()

        detector.performance_report()

        detector.summary()

        logger.info("=" * 70)
        logger.info("YOLOv8 FACE DETECTOR COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as error:

        logger.exception(
            f"Pipeline Failed : {error}"
        )