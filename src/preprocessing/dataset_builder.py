"""
==========================================================
Project : Anime Character Persona & Sentiment Engine
Module  : dataset_builder.py
Phase   : Phase 1 - Dataset Builder

Description
-----------
Creates the final multimodal dataset by combining
NLP features and Computer Vision statistics.

Author : Puviarasan
==========================================================
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config.paths import (
    FEATURE_ENGINEERED_CSV,
    IMAGE_STATISTICS_CSV,
    PROCESSED_IMAGE,
    FINAL_DATASET_CSV,
    DATASET_REPORT,
)

# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# NumPy JSON Converter
# ==========================================================

def convert_numpy(obj: Any):
    """
    Converts NumPy data types into native Python types
    so they can be saved as JSON.
    """

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    raise TypeError(
        f"Object of type {type(obj)} is not JSON serializable."
    )

# ==========================================================
# Dataset Builder
# ==========================================================

class DatasetBuilder:

    """
    Builds the final multimodal dataset.
    """

    def __init__(self):

        self.dialogue_df: pd.DataFrame | None = None

        self.image_df: pd.DataFrame | None = None

        self.final_df: pd.DataFrame | None = None

    # ======================================================
    # Load Dialogue Dataset
    # ======================================================

    def load_dialogue_dataset(self):

        logger.info("Loading dialogue dataset...")

        self.dialogue_df = pd.read_csv(
            FEATURE_ENGINEERED_CSV
        )

        logger.info(
            f"Dialogue rows : {len(self.dialogue_df)}"
        )

    # ======================================================
    # Load Image Dataset
    # ======================================================

    def load_image_dataset(self):

        logger.info("Loading image statistics...")

        self.image_df = pd.read_csv(
            IMAGE_STATISTICS_CSV
        )

        logger.info(
            f"Image rows : {len(self.image_df)}"
        )

    # ======================================================
    # Validate Tensor
    # ======================================================

    def validate_tensor(self) -> bool:

        if PROCESSED_IMAGE.exists():

            logger.info(
                "Tensor file found."
            )

            return True

        logger.warning(
            "Tensor file missing."
        )

        return False

    # ======================================================
    # Validate Dialogue Dataset
    # ======================================================

    def validate_dialogue_dataset(self) -> Dict:

        logger.info(
            "Validating dialogue dataset..."
        )

        report = {

            "rows": int(
                len(self.dialogue_df)
            ),

            "columns": int(
                len(self.dialogue_df.columns)
            ),

            "missing_values": int(
                self.dialogue_df.isnull().sum().sum()
            ),

            "duplicate_rows": int(
                self.dialogue_df.duplicated().sum()
            )

        }

        print("\n")
        print("=" * 60)
        print("DIALOGUE DATASET")
        print("=" * 60)

        for key, value in report.items():
            print(f"{key:20}: {value}")

        print("=" * 60)
        print()

        return report

    # ======================================================
    # Validate Image Dataset
    # ======================================================

    def validate_image_dataset(self) -> Dict:

        logger.info(
            "Validating image dataset..."
        )

        report = {

            "rows": int(
                len(self.image_df)
            ),

            "columns": int(
                len(self.image_df.columns)
            ),

            "missing_values": int(
                self.image_df.isnull().sum().sum()
            ),

            "duplicate_rows": int(
                self.image_df.duplicated().sum()
            )

        }

        print("\n")
        print("=" * 60)
        print("IMAGE DATASET")
        print("=" * 60)

        for key, value in report.items():
            print(f"{key:20}: {value}")

        print("=" * 60)
        print()

        return report
        # ======================================================
    # Merge Datasets
    # ======================================================

    def merge_datasets(self) -> pd.DataFrame:

        logger.info("Merging datasets...")

        image_info = self.image_df.iloc[0].to_dict()

        self.final_df = self.dialogue_df.copy()

        # Add image statistics to every dialogue row
        for column, value in image_info.items():

            self.final_df[column] = value

        # Add tensor path
        self.final_df["tensor_path"] = str(PROCESSED_IMAGE)

        logger.info("Datasets merged successfully.")

        return self.final_df

    # ======================================================
    # Character Analytics
    # ======================================================

    def character_analytics(self) -> pd.DataFrame:

        logger.info(
            "Generating character analytics..."
        )

        analytics = (

            self.final_df

            .groupby("character")

            .agg(

                total_dialogues=("character", "count"),

                average_words=("word_count", "mean"),

                average_emotion=("emotion_intensity", "mean"),

                average_duration=("duration_seconds", "mean"),

                average_speed=("words_per_second", "mean")

            )

            .round(2)

            .reset_index()

        )

        print("\n")
        print("=" * 70)
        print("CHARACTER ANALYTICS")
        print("=" * 70)
        print(analytics)
        print("=" * 70)
        print()

        return analytics

    # ======================================================
    # Feature Statistics
    # ======================================================

    def feature_statistics(self) -> pd.DataFrame:

        logger.info(
            "Generating feature statistics..."
        )

        numeric = self.final_df.select_dtypes(
            include="number"
        )

        stats = numeric.describe().T

        print("\n")
        print("=" * 70)
        print("FEATURE STATISTICS")
        print("=" * 70)
        print(stats)
        print("=" * 70)
        print()

        return stats

    # ======================================================
    # Dataset Statistics
    # ======================================================

    def dataset_statistics(self) -> dict:

        logger.info(
            "Generating dataset statistics..."
        )

        report = {

            "rows": int(len(self.final_df)),

            "columns": int(
                len(self.final_df.columns)
            ),

            "numeric_columns": int(

                len(

                    self.final_df.select_dtypes(

                        include="number"

                    ).columns

                )

            ),

            "categorical_columns": int(

                len(

                    self.final_df.select_dtypes(

                        exclude="number"

                    ).columns

                )

            ),

            "memory_kb": round(

                self.final_df.memory_usage(

                    deep=True

                ).sum() / 1024,

                2

            )

        }

        print("\n")
        print("=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)

        for key, value in report.items():

            print(f"{key:25}: {value}")

        print("=" * 70)
        print()

        return report

    # ======================================================
    # Missing Value Summary
    # ======================================================

    def missing_value_summary(self) -> pd.DataFrame:

        missing = (

            self.final_df

            .isnull()

            .sum()

            .reset_index()

        )

        missing.columns = [

            "Column",

            "Missing Values"

        ]

        print("\n")
        print("=" * 70)
        print("MISSING VALUE SUMMARY")
        print("=" * 70)
        print(missing)
        print("=" * 70)
        print()

        return missing

    # ======================================================
    # Duplicate Summary
    # ======================================================

    def duplicate_summary(self):

        duplicates = int(

            self.final_df.duplicated().sum()

        )

        print("\n")
        print("=" * 70)
        print("DUPLICATE ROWS")
        print("=" * 70)
        print(duplicates)
        print("=" * 70)
        print()

        return duplicates
        # ======================================================
    # ML Readiness Score
    # ======================================================

    def ml_readiness_score(self) -> float:

        score = 100.0

        # Missing value penalty
        missing = int(
            self.final_df.isnull().sum().sum()
        )

        score -= missing

        # Duplicate penalty
        duplicates = int(
            self.final_df.duplicated().sum()
        )

        score -= duplicates

        # Tensor penalty
        if not PROCESSED_IMAGE.exists():

            score -= 10

        score = max(score, 0)

        score = round(score, 2)

        print("\n")
        print("=" * 70)
        print(f"ML READINESS SCORE : {score}%")
        print("=" * 70)
        print()

        return score

    # ======================================================
    # Dataset Health Score
    # ======================================================

    def dataset_health_score(self) -> float:

        total_cells = (
            self.final_df.shape[0] *
            self.final_df.shape[1]
        )

        missing = int(
            self.final_df.isnull().sum().sum()
        )

        completeness = (
            (total_cells - missing) /
            total_cells
        ) * 40

        # Image Quality (30%)

        if "quality_score" in self.final_df.columns:

            image_quality = (

                self.final_df[
                    "quality_score"
                ].mean()

                / 100

            ) * 30

        else:

            image_quality = 30

        # Dialogue Quality (30%)

        if "emotion_intensity" in self.final_df.columns:

            dialogue_quality = (

                self.final_df[
                    "emotion_intensity"
                ].mean()

                / 100

            ) * 30

        else:

            dialogue_quality = 30

        health = (

            completeness +

            image_quality +

            dialogue_quality

        )

        health = round(

            min(health, 100),

            2

        )

        print("\n")
        print("=" * 70)
        print(f"DATASET HEALTH SCORE : {health}%")
        print("=" * 70)
        print()

        return health

    # ======================================================
    # Save Final Dataset
    # ======================================================

    def save_dataset(self):

        logger.info(
            "Saving final dataset..."
        )

        self.final_df.to_csv(

            FINAL_DATASET_CSV,

            index=False

        )

        logger.info(

            f"Saved:\n{FINAL_DATASET_CSV}"

        )

    # ======================================================
    # Generate JSON Report
    # ======================================================

    def generate_report(self):

        report = {

            "rows": int(
                len(self.final_df)
            ),

            "columns": int(
                len(self.final_df.columns)
            ),

            "numeric_columns": int(

                len(

                    self.final_df.select_dtypes(

                        include="number"

                    ).columns

                )

            ),

            "categorical_columns": int(

                len(

                    self.final_df.select_dtypes(

                        exclude="number"

                    ).columns

                )

            ),

            "missing_values": int(

                self.final_df

                .isnull()

                .sum()

                .sum()

            ),

            "duplicate_rows": int(

                self.final_df

                .duplicated()

                .sum()

            ),

            "ml_readiness": float(

                self.ml_readiness_score()

            ),

            "dataset_health": float(

                self.dataset_health_score()

            ),

            "tensor_exists": bool(

                PROCESSED_IMAGE.exists()

            ),

            "ready_for_training": True

        }

        with open(

            DATASET_REPORT,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                report,

                file,

                indent=4,

                default=convert_numpy

            )

        logger.info(

            f"Saved:\n{DATASET_REPORT}"

        )

        return report

    # ======================================================
    # Dataset Summary
    # ======================================================

    def summary(self):

        print("\n")

        print("=" * 70)

        print("FINAL MULTIMODAL DATASET")

        print("=" * 70)

        print(self.final_df.head())

        print()

        print(

            f"Rows    : {len(self.final_df)}"

        )

        print(

            f"Columns : {len(self.final_df.columns)}"

        )

        print("=" * 70)

        print()

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("Anime Character Persona Engine")
    logger.info("Phase 1 - Dataset Builder")
    logger.info("=" * 70)

    builder = DatasetBuilder()

    try:

        # --------------------------------------------------
        # Load Datasets
        # --------------------------------------------------

        builder.load_dialogue_dataset()

        builder.load_image_dataset()

        builder.validate_tensor()

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        builder.validate_dialogue_dataset()

        builder.validate_image_dataset()

        # --------------------------------------------------
        # Merge
        # --------------------------------------------------

        builder.merge_datasets()

        # --------------------------------------------------
        # Analytics
        # --------------------------------------------------

        builder.character_analytics()

        builder.feature_statistics()

        builder.dataset_statistics()

        builder.missing_value_summary()

        builder.duplicate_summary()

        builder.ml_readiness_score()

        builder.dataset_health_score()

        # --------------------------------------------------
        # Save Outputs
        # --------------------------------------------------

        builder.save_dataset()

        builder.generate_report()

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        builder.summary()

        logger.info("=" * 70)
        logger.info("PHASE 1 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as error:

        logger.exception(

            f"Dataset Builder Failed : {error}"

        )