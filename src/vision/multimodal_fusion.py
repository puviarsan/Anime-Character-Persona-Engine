"""
==========================================================
Project : Anime Character Persona Engine

Module  : multimodal_fusion.py

Description
-----------
Phase 4

Multimodal Fusion Engine

1. Load Image Embeddings
2. Load Sentiment Results
3. Fuse Vision + NLP Features
4. Save Multimodal Embeddings
==========================================================
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.paths import (

    DATA_DIR,

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    ANALYTICS_DATA_DIR,

    SAMPLE_DATA_DIR,

    RAW_SUBTITLE_DIR,

    RAW_IMAGE_DIR,

    RAW_VIDEO_DIR,

    PROCESSED_IMAGE_DIR,

    MODEL_DIR,

    EMBEDDINGS_DIR,

    FUSED_DIR,

    SENTIMENT_RESULTS_CSV,

    MULTIMODAL_FEATURES_FILE

)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Multimodal Fusion
# ==========================================================

class MultimodalFusion:

    def __init__(self):

        self.embeddings = []

        self.sentiments = None

        self.fused_results = []

        self.start_time = 0

        self.end_time = 0


    # ======================================================
    # Load Embeddings
    # ======================================================

    def load_embeddings(self):

        logger.info(
            "Loading image embeddings..."
        )

        self.embeddings = []

        for character_dir in EMBEDDINGS_DIR.iterdir():

            if not character_dir.is_dir():

                continue

            character = character_dir.name

            for embedding_file in character_dir.glob("*.npy"):

                embedding = np.load(
                    embedding_file
                )

                self.embeddings.append(

                    {
                        "character": character,
                        "filename": embedding_file.stem,
                        "embedding": embedding
                    }

                )

        logger.info(
            f"Embeddings Loaded : {len(self.embeddings)}"
        )


    # ======================================================
    # Load Sentiment
    # ======================================================

    def load_sentiments(self):

        logger.info(
            "Loading sentiment dataset..."
        )

        self.sentiments = pd.read_csv(
            SENTIMENT_RESULTS_CSV
        )

        logger.info(
            f"Dialogues Loaded : {len(self.sentiments)}"
        )


    # ======================================================
    # Validation
    # ======================================================

    def validate(self):

        if len(self.embeddings) == 0:

            raise ValueError(
                "No embeddings found."
            )

        if self.sentiments.empty:

            raise ValueError(
                "Sentiment dataset is empty."
            )

        logger.info(
            "Validation successful."
        )


    # ======================================================
    # Match Sentiment
    # ======================================================

    def get_sentiment_vector(
        self,
        character
    ):

        dataframe = self.sentiments[
            self.sentiments["character"].str.lower()
            ==
            character.lower()
        ]

        if dataframe.empty:

            return np.zeros(7)

        return np.array(

            [

                dataframe["positive"].mean(),

                dataframe["negative"].mean(),

                dataframe["neutral"].mean(),

                dataframe["compound"].mean(),

                dataframe["anime_score"].mean(),

                dataframe["hybrid_score"].mean(),

                dataframe["confidence"].mean()

            ],

            dtype=np.float32

        )


    # ======================================================
    # Feature Fusion
    # ======================================================

    def fuse_features(self):

        logger.info(
            "Creating multimodal vectors..."
        )

        self.start_time = time.time()

        self.fused_results = []

        for item in self.embeddings:

            image_vector = item["embedding"]

            sentiment_vector = self.get_sentiment_vector(

                item["character"]

            )

            fused_vector = np.concatenate(

                [

                    image_vector,

                    sentiment_vector

                ]

            )

            self.fused_results.append(

                {

                    "character": item["character"],

                    "filename": item["filename"],

                    "vector": fused_vector

                }

            )

        logger.info(

            f"Fused Samples : {len(self.fused_results)}"

        )
        # ======================================================
    # Save Fused Embeddings
    # ======================================================

    def save_embeddings(self):

        logger.info(
            "Saving fused embeddings..."
        )

        for item in self.fused_results:

            character_dir = (

                FUSED_DIR /

                item["character"]

            )

            character_dir.mkdir(

                parents=True,

                exist_ok=True

            )

            output_file = (

                character_dir /

                f"{item['filename']}.npy"

            )

            np.save(

                output_file,

                item["vector"]

            )

            item["path"] = str(

                output_file

            )

        logger.info(

            "Fused embeddings saved successfully."

        )


    # ======================================================
    # Save CSV Report
    # ======================================================

    def save_report(self):

        logger.info(

            "Saving multimodal report..."

        )

        rows = []

        for item in self.fused_results:

            vector = item["vector"]

            rows.append(

                {

                    "character": item["character"],

                    "filename": item["filename"],

                    "vector_length": len(vector),

                    "mean": round(

                        float(np.mean(vector)),

                        4

                    ),

                    "std": round(

                        float(np.std(vector)),

                        4

                    ),

                    "min": round(

                        float(np.min(vector)),

                        4

                    ),

                    "max": round(

                        float(np.max(vector)),

                        4

                    ),

                    "embedding_path": item["path"]

                }

            )

        dataframe = pd.DataFrame(

            rows

        )

        dataframe.to_csv(

            MULTIMODAL_FEATURES_FILE,

            index=False

        )

        logger.info(

            f"Saved:\n{MULTIMODAL_FEATURES_FILE}"

        )

        return dataframe


    # ======================================================
    # Dataset Statistics
    # ======================================================

    def dataset_statistics(self):

        print()

        print("=" * 70)

        print("MULTIMODAL DATASET")

        print("=" * 70)

        print(

            f"Characters : {len(set(item['character'] for item in self.fused_results))}"

        )

        print(

            f"Samples    : {len(self.fused_results)}"

        )

        print(

            f"Vector Size: {len(self.fused_results[0]['vector'])}"

        )

        print("=" * 70)

        print()


    # ======================================================
    # Performance
    # ======================================================

    def performance(self):

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

        dataframe = pd.DataFrame(

            {

                "mean": [

                    np.mean(

                        item["vector"]

                    )

                    for item in self.fused_results

                ],

                "std": [

                    np.std(

                        item["vector"]

                    )

                    for item in self.fused_results

                ]

            }

        )

        print()

        print("=" * 70)

        print("MULTIMODAL FUSION SUMMARY")

        print("=" * 70)

        print(

            f"Total Samples     : {len(self.fused_results)}"

        )

        print(

            f"Characters        : {len(set(item['character'] for item in self.fused_results))}"

        )

        print(

            f"Vector Length     : {len(self.fused_results[0]['vector'])}"

        )

        print(

            f"Average Mean      : {round(float(dataframe['mean'].mean()),4)}"

        )

        print(

            f"Average Std       : {round(float(dataframe['std'].mean()),4)}"

        )

        print("=" * 70)

        print()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    logger.info("=" * 70)

    logger.info(
        "Anime Character Persona Engine"
    )

    logger.info(
        "Phase 4 - Multimodal Fusion"
    )

    logger.info("=" * 70)

    fusion = MultimodalFusion()

    try:

        fusion.load_embeddings()

        fusion.load_sentiments()

        fusion.validate()

        fusion.fuse_features()

        fusion.save_embeddings()

        fusion.save_report()

        fusion.dataset_statistics()

        fusion.performance()

        fusion.summary()

        logger.info("=" * 70)

        logger.info(
            "MULTIMODAL FUSION COMPLETED SUCCESSFULLY"
        )

        logger.info("=" * 70)

    except Exception as error:

        logger.exception(

            f"Pipeline Failed : {error}"

        )