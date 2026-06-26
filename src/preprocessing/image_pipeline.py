"""
==========================================================
Project : Anime Character Persona & Sentiment Engine

Module  : image_pipeline.py

Phase   : Phase 1

Description
-----------
Loads anime images and prepares them
for Deep Learning.
==========================================================
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config.paths import (
    SAMPLE_IMAGE,
    PROCESSED_IMAGE,
    IMAGE_STATISTICS_CSV,
    IMAGE_QUALITY_REPORT
)

# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


# ==========================================================
# Image Pipeline
# ==========================================================

class ImagePipeline:

    def __init__(self, image_path):

        self.image_path = Path(image_path)

        self.image = None

        self.tensor = None

        self.processing_time = {}

    # ------------------------------------------------------

    def validate_image(self):

        if not self.image_path.exists():

            raise FileNotFoundError(

                f"Image not found:\n{self.image_path}"

            )

        if self.image_path.suffix.lower() not in [

            ".jpg",

            ".jpeg",

            ".png"

        ]:

            raise ValueError(

                "Only JPG, JPEG and PNG supported."

            )

        logging.info("Image validation successful.")

    # ------------------------------------------------------

    def load_image(self):

        start = time.time()

        self.image = Image.open(

            self.image_path

        ).convert("RGB")

        self.processing_time["Load Image"] = round(

            time.time() - start,

            4

        )

        logging.info("Image loaded successfully.")

        return self.image

    # ------------------------------------------------------

    def image_information(self):

        width, height = self.image.size

        channels = len(

            self.image.getbands()

        )

        print("\n")

        print("=" * 60)

        print("IMAGE INFORMATION")

        print("=" * 60)

        print(f"Width      : {width}")

        print(f"Height     : {height}")

        print(f"Channels   : {channels}")

        print(f"Format     : {self.image.format}")

        print(f"Mode       : {self.image.mode}")

        print("=" * 60)

        print()

        return {

            "width": width,

            "height": height,

            "channels": channels

        }

    # ------------------------------------------------------

    @staticmethod
    def pil_to_numpy(image):

        return np.array(image)
        # ------------------------------------------------------
    # Image Quality Analysis
    # ------------------------------------------------------

    def brightness(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return round(float(np.mean(gray)), 2)

    # ------------------------------------------------------

    def contrast(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return round(float(np.std(gray)), 2)

    # ------------------------------------------------------

    def blur_score(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        return round(float(score), 2)

    # ------------------------------------------------------

    def resolution_score(self):

        width, height = self.image.size

        pixels = width * height

        if pixels >= 1920 * 1080:
            return 100

        elif pixels >= 1280 * 720:
            return 80

        elif pixels >= 640 * 480:
            return 60

        else:
            return 40

    # ------------------------------------------------------

    def quality_score(
        self,
        brightness,
        contrast,
        blur,
        resolution
    ):

        score = (

            brightness * 0.25 +

            contrast * 0.20 +

            min(blur, 100) * 0.30 +

            resolution * 0.25

        )

        return round(score, 2)

    # ------------------------------------------------------

    def analyze_quality(self):

        logging.info(

            "Analyzing image quality..."

        )

        image = self.pil_to_numpy(

            self.image

        )

        brightness = self.brightness(image)

        contrast = self.contrast(image)

        blur = self.blur_score(image)

        resolution = self.resolution_score()

        quality = self.quality_score(

            brightness,

            contrast,

            blur,

            resolution

        )

        report = {

            "brightness": brightness,

            "contrast": contrast,

            "blur_score": blur,

            "resolution_score": resolution,

            "quality_score": quality

        }

        print("\n")

        print("=" * 60)

        print("IMAGE QUALITY")

        print("=" * 60)

        for key, value in report.items():

            print(f"{key:20}: {value}")

        print("=" * 60)

        print()

        return report

    # ------------------------------------------------------
    # Image Transform
    # ------------------------------------------------------

    def preprocess_image(self):

        logging.info(

            "Preprocessing image..."

        )

        start = time.time()

        transform = transforms.Compose([

            transforms.Resize(

                (224, 224)

            ),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[

                    0.485,

                    0.456,

                    0.406

                ],

                std=[

                    0.229,

                    0.224,

                    0.225

                ]

            )

        ])

        tensor = transform(

            self.image

        )

        tensor = tensor.unsqueeze(0)

        self.processing_time[

            "Preprocessing"

        ] = round(

            time.time() - start,

            4

        )

        self.tensor = tensor

        logging.info(

            "Tensor created successfully."

        )

        return tensor

    # ------------------------------------------------------

    def tensor_statistics(self):

        tensor = self.tensor

        stats = {

            "shape": list(tensor.shape),

            "mean": round(

                float(tensor.mean()),

                4

            ),

            "std": round(

                float(tensor.std()),

                4

            ),

            "min": round(

                float(tensor.min()),

                4

            ),

            "max": round(

                float(tensor.max()),

                4

            )

        }

        print("\n")

        print("=" * 60)

        print("TENSOR STATISTICS")

        print("=" * 60)

        for key, value in stats.items():

            print(f"{key:15}: {value}")

        print("=" * 60)

        print()

        return stats
        # ------------------------------------------------------
    # Save Tensor
    # ------------------------------------------------------

    def save_tensor(self):

        torch.save(

            self.tensor,

            PROCESSED_IMAGE

        )

        logging.info(

            f"Tensor saved:\n{PROCESSED_IMAGE}"

        )

    # ------------------------------------------------------
    # Save Image Statistics
    # ------------------------------------------------------

    def save_statistics(

        self,

        image_info,

        tensor_stats

    ):

        dataframe = pd.DataFrame([{

            **image_info,

            **tensor_stats

        }])

        dataframe.to_csv(

            IMAGE_STATISTICS_CSV,

            index=False

        )

        logging.info(

            f"Image statistics saved:\n"

            f"{IMAGE_STATISTICS_CSV}"

        )

    # ------------------------------------------------------
    # Save Quality Report
    # ------------------------------------------------------

    def save_quality_report(

        self,

        report

    ):

        with open(

            IMAGE_QUALITY_REPORT,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                report,

                file,

                indent=4

            )

        logging.info(

            f"Quality report saved:\n"

            f"{IMAGE_QUALITY_REPORT}"

        )

    # ------------------------------------------------------
    # Performance Report
    # ------------------------------------------------------

    def performance_report(self):

        print("\n")

        print("=" * 60)

        print("PIPELINE PERFORMANCE")

        print("=" * 60)

        total = 0

        for name, value in self.processing_time.items():

            total += value

            print(

                f"{name:20}: {value:.4f} sec"

            )

        print("-" * 60)

        print(

            f"{'Total':20}: {total:.4f} sec"

        )

        print("=" * 60)

        print()
# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    logging.info(

        "Starting Image Pipeline..."

    )

    pipeline = ImagePipeline(

        SAMPLE_IMAGE

    )

    pipeline.validate_image()

    pipeline.load_image()

    image_info = pipeline.image_information()

    quality = pipeline.analyze_quality()

    pipeline.preprocess_image()

    tensor_stats = pipeline.tensor_statistics()

    pipeline.save_tensor()

    pipeline.save_statistics(

        image_info,

        tensor_stats

    )

    pipeline.save_quality_report(

        quality

    )

    pipeline.performance_report()

    logging.info(

        "Image Pipeline Completed Successfully."

    )