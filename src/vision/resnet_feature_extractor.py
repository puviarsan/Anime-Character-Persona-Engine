"""
==========================================================
Project : Anime Character Persona Engine

Module  : resnet_feature_extractor.py

Phase   : Phase 3

Description
-----------
ResNet18 Feature Extraction

1. Load Cropped Faces
2. Preprocess Images
3. Extract 512-D Features
4. Save Embeddings
==========================================================
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image

from config.paths import (
    FACE_OUTPUT_DIR,
    ANALYTICS_DATA_DIR,
    PROCESSED_DATA_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# ResNet Feature Extractor
# ==========================================================

class ResNetFeatureExtractor:

    """
    Extract 512-D visual embeddings
    from cropped anime faces.
    """

    def __init__(self):

        self.dataset = []

        self.results = []

        self.model = None

        self.transform = None

        self.device = torch.device("cpu")

        self.start_time = 0

        self.end_time = 0

        logger.info(
            "Loading ResNet18..."
        )

        self.load_model()

        self.build_transform()

        logger.info(
            "ResNet18 Loaded Successfully."
        )
    
        # ======================================================
    # Load ResNet18
    # ======================================================

    def load_model(self):

        weights = models.ResNet18_Weights.DEFAULT

        network = models.resnet18(
            weights=weights
        )

        network.fc = nn.Identity()

        network.eval()

        network.to(
            self.device
        )

        self.model = network

        # ======================================================
    # Image Transform
    # ======================================================

    def build_transform(self):

        self.transform = transforms.Compose(

            [

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

            ]

        )

        # ======================================================
    # Scan Face Dataset
    # ======================================================

    def scan_dataset(self):

        logger.info(
            "Scanning cropped faces..."
        )

        self.dataset = []

        extensions = (

            ".jpg",

            ".jpeg",

            ".png"

        )

        for character_folder in FACE_OUTPUT_DIR.iterdir():

            if not character_folder.is_dir():

                continue

            if character_folder.name == "detections":

                continue

            character = character_folder.name

            for image_path in character_folder.iterdir():

                if image_path.suffix.lower() not in extensions:

                    continue

                self.dataset.append(

                    {

                        "character": character,

                        "image_path": image_path

                    }

                )

        logger.info(

            f"Faces Found : {len(self.dataset)}"

        )

        return self.dataset
        # ======================================================
    # Validate Dataset
    # ======================================================

    def validate_dataset(self):

        if len(self.dataset) == 0:

            raise ValueError(

                "No cropped faces found."

            )

        logger.info(

            "Dataset validation successful."

        )

        # ======================================================
    # Load Image
    # ======================================================

    def load_image(self, image_info):

        image_path = image_info["image_path"]

        logger.info(

            f"Loading : {image_path.name}"

        )

        image = Image.open(

            image_path

        ).convert(

            "RGB"

        )

        tensor = self.transform(

            image

        )

        tensor = tensor.unsqueeze(

            0

        )

        tensor = tensor.to(

            self.device

        )

        return tensor
        # ======================================================
    # Extract Features
    # ======================================================

    def extract_features(

        self,

        tensor

    ):

        with torch.no_grad():

            features = self.model(

                tensor

            )

        embedding = features.squeeze(

            0

        ).cpu().numpy()

        return embedding
        # ======================================================
    # Feature Statistics
    # ======================================================

    def feature_statistics(

        self,

        embedding,

        image_info

    ):

        print()

        print("=" * 70)

        print(

            f"FEATURE STATISTICS : {image_info['image_path'].name}"

        )

        print("=" * 70)

        print(

            f"Character : {image_info['character']}"

        )

        print(

            f"Vector Length : {len(embedding)}"

        )

        print(

            f"Mean : {round(float(np.mean(embedding)),4)}"

        )

        print(

            f"Std : {round(float(np.std(embedding)),4)}"

        )

        print(

            f"Min : {round(float(np.min(embedding)),4)}"

        )

        print(

            f"Max : {round(float(np.max(embedding)),4)}"

        )

        print("=" * 70)

        print()

        # ======================================================
    # Process One Image
    # ======================================================

    def process_image(

        self,

        image_info

    ):

        tensor = self.load_image(

            image_info

        )

        embedding = self.extract_features(

            tensor

        )

        self.feature_statistics(

            embedding,

            image_info

        )

        self.results.append(

            {

                "character": image_info["character"],

                "filename": image_info["image_path"].name,

                "embedding": embedding

            }

        )
        # ======================================================
    # Process Dataset
    # ======================================================

    def process_dataset(self):

        logger.info(

            "Extracting ResNet18 features..."

        )

        self.results = []

        self.start_time = time.time()

        for image_info in self.dataset:

            self.process_image(

                image_info

            )

        logger.info(

            f"Embeddings Generated : {len(self.results)}"

        )

        return self.results
        # ======================================================
    # Save Embeddings
    # ======================================================

    def save_embeddings(self):

        logger.info(
            "Saving embeddings..."
        )

        embedding_dir = (

            PROCESSED_DATA_DIR /

            "embeddings"

        )

        embedding_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        for result in self.results:

            character_dir = (

                embedding_dir /

                result["character"]

            )

            character_dir.mkdir(

                parents=True,

                exist_ok=True

            )

            output_file = (

                character_dir /

                f"{Path(result['filename']).stem}.npy"

            )

            np.save(

                output_file,

                result["embedding"]

            )

            result["embedding_path"] = str(

                output_file

            )

        logger.info(

            "Embeddings saved successfully."

        )

        # ======================================================
    # Save Feature Report
    # ======================================================

    def save_report(self):

        logger.info(

            "Saving feature report..."

        )

        rows = []

        for result in self.results:

            embedding = result["embedding"]

            rows.append(

                {

                    "character": result["character"],

                    "filename": result["filename"],

                    "vector_length": len(embedding),

                    "mean": round(

                        float(np.mean(embedding)),

                        4

                    ),

                    "std": round(

                        float(np.std(embedding)),

                        4

                    ),

                    "min": round(

                        float(np.min(embedding)),

                        4

                    ),

                    "max": round(

                        float(np.max(embedding)),

                        4

                    ),

                    "embedding_path": result["embedding_path"]

                }

            )

        dataframe = pd.DataFrame(

            rows

        )

        report_file = (

            ANALYTICS_DATA_DIR /

            "resnet_features.csv"

        )

        dataframe.to_csv(

            report_file,

            index=False

        )

        logger.info(

            f"Saved:\n{report_file}"

        )

        return dataframe
        # ======================================================
    # Dataset Statistics
    # ======================================================

    def dataset_statistics(self):

        print()

        print("=" * 70)

        print("RESNET18 FEATURE DATASET")

        print("=" * 70)

        print(

            f"Images Processed : {len(self.results)}"

        )

        print(

            f"Characters : {len(set(item['character'] for item in self.results))}"

        )

        print(

            "Embedding Size : 512"

        )

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

        print(

            f"Execution Time : {elapsed} sec"

        )

        print("=" * 70)

        print()

        # ======================================================
    # Summary
    # ======================================================

    def summary(self):

        print()

        print("=" * 70)
        print("RESNET18 FEATURE EXTRACTION SUMMARY")
        print("=" * 70)

        print(f"Total Images      : {len(self.results)}")
        print(f"Total Characters  : {len(set(item['character'] for item in self.results))}")
        print("Embedding Length  : 512")

        if len(self.results) > 0:

            dataframe = pd.DataFrame({

                "mean": [
                    float(np.mean(item["embedding"]))
                    for item in self.results
                ],

                "std": [
                    float(np.std(item["embedding"]))
                    for item in self.results
                ]

            })

            print(f"Average Mean      : {round(dataframe['mean'].mean(),4)}")
            print(f"Average Std       : {round(dataframe['std'].mean(),4)}")

        else:

            print("Average Mean      : N/A")
            print("Average Std       : N/A")

        print("=" * 70)
        print()

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("Anime Character Persona Engine")
    logger.info("Phase 3 - ResNet18 Feature Extractor")
    logger.info("=" * 70)

    extractor = ResNetFeatureExtractor()

    try:

        extractor.scan_dataset()

        extractor.validate_dataset()

        extractor.process_dataset()

        extractor.save_embeddings()

        extractor.save_report()

        extractor.dataset_statistics()

        extractor.performance_report()

        extractor.summary()

        logger.info("=" * 70)
        logger.info("RESNET18 FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as error:

        logger.exception(

            f"Pipeline Failed : {error}"

        )