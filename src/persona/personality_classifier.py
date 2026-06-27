"""
==========================================================
Project : Anime Character Persona Engine

Module  : personality_classifier.py

Description
-----------
Phase 5

Personality Classification Engine

1. Load Multimodal Dataset
2. Load Fused Embeddings
3. Predict Personality Traits
4. Detect Archetype
5. Generate Personality Report
==========================================================
"""

from __future__ import annotations

import logging
import time

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

    MULTIMODAL_FEATURES_FILE,

    PERSONALITY_REPORT_FILE

     

)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class PersonalityClassifier:

    """
    Anime Character Personality Classifier
    """

    def __init__(self):

        self.multimodal_dataframe = None

        self.dataset = []

        self.results = []

        self.start_time = 0

        self.end_time = 0


    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        logger.info(
            "Loading multimodal dataset..."
        )

        self.multimodal_dataframe = pd.read_csv(
            MULTIMODAL_FEATURES_FILE
        )

        logger.info(
            f"Rows Loaded : {len(self.multimodal_dataframe)}"
        )


    # ======================================================
    # Load Embeddings
    # ======================================================

    def load_embeddings(self):

        logger.info(
            "Loading fused embeddings..."
        )

        self.dataset = []

        for character_folder in FUSED_DIR.iterdir():

            if not character_folder.is_dir():

                continue

            character = character_folder.name

            for embedding_file in character_folder.glob("*.npy"):

                embedding = np.load(
                    embedding_file
                )

                self.dataset.append(

                    {
                        "character": character,
                        "filename": embedding_file.stem,
                        "embedding": embedding
                    }

                )

        logger.info(
            f"Embeddings Loaded : {len(self.dataset)}"
        )


    # ======================================================
    # Validation
    # ======================================================

    def validate(self):

        if self.multimodal_dataframe is None:

            raise ValueError(
                "Multimodal dataset not loaded."
            )

        if len(self.dataset) == 0:

            raise ValueError(
                "No fused embeddings found."
            )

        logger.info(
            "Validation successful."
        )


    # ======================================================
    # Calculate Personality Scores
    # ======================================================

    def personality_scores(
        self,
        embedding
    ):

        mean = float(np.mean(embedding))
        std = float(np.std(embedding))
        maximum = float(np.max(embedding))
        minimum = float(np.min(embedding))

        leadership = min(100, round(mean * 95, 2))

        confidence = min(100, round(maximum * 18, 2))

        optimism = min(100, round(mean * 105, 2))

        kindness = max(0, min(100, round(100 - std * 12, 2)))

        aggression = max(0, min(100, round(std * 30, 2)))

        emotional_stability = max(
            0,
            min(
                100,
                round(100 - abs(minimum) * 25, 2)
            )
        )

        intelligence = min(
            100,
            round((mean + std) * 50, 2)
        )

        bravery = min(
            100,
            round((maximum / 6.5) * 100, 2)
        )

        teamwork = max(
            0,
            min(
                100,
                round(100 - aggression * 0.4, 2)
            )
        )

        discipline = max(
            0,
            min(
                100,
                round((leadership + emotional_stability) / 2, 2)
            )
        )

        return {

            "leadership": leadership,

            "confidence": confidence,

            "optimism": optimism,

            "kindness": kindness,

            "aggression": aggression,

            "emotional_stability": emotional_stability,

            "intelligence": intelligence,

            "bravery": bravery,

            "teamwork": teamwork,

            "discipline": discipline

        }


    # ======================================================
    # Detect Archetype
    # ======================================================

    def detect_archetype(
        self,
        scores
    ):

        if (
            scores["leadership"] >= 90 and
            scores["optimism"] >= 90
        ):

            return "Hero"

        elif (
            scores["aggression"] >= 70
        ):

            return "Rival"

        elif (
            scores["intelligence"] >= 85 and
            scores["discipline"] >= 80
        ):

            return "Strategist"

        elif (
            scores["kindness"] >= 90
        ):

            return "Protector"

        elif (
            scores["emotional_stability"] >= 90
        ):

            return "Mentor"

        elif (
            scores["confidence"] >= 90
        ):

            return "Commander"

        else:

            return "Anti-Hero"


    # ======================================================
    # Predict Personality
    # ======================================================

    def predict(self):

        logger.info(
            "Predicting personalities..."
        )

        self.start_time = time.time()

        self.results = []

        for sample in self.dataset:

            scores = self.personality_scores(
                sample["embedding"]
            )

            archetype = self.detect_archetype(
                scores
            )

            self.results.append(

                {

                    "character": sample["character"],

                    "filename": sample["filename"],

                    "archetype": archetype,

                    **scores

                }

            )

        logger.info(
            f"Predictions Generated : {len(self.results)}"
        )
        # ======================================================
    # Save Personality Report
    # ======================================================

    def save_report(self):

        logger.info(
            "Saving personality report..."
        )

        dataframe = pd.DataFrame(
            self.results
        )

        report_file = (
            ANALYTICS_DATA_DIR /
            "personality_report.csv"
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
        print("PERSONALITY DATASET")
        print("=" * 70)

        print(
            f"Characters : {self.multimodal_dataframe['character'].nunique()}"
        )

        print(
            f"Samples    : {len(self.results)}"
        )

        print(
            f"Archetypes : {len(set(item['archetype'] for item in self.results))}"
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
            self.results
        )

        print()
        print("=" * 70)
        print("PERSONALITY CLASSIFICATION SUMMARY")
        print("=" * 70)

        print(
            f"Total Samples      : {len(dataframe)}"
        )

        print(
            f"Characters         : {dataframe['character'].nunique()}"
        )

        print(
            f"Hero               : {(dataframe['archetype']=='Hero').sum()}"
        )

        print(
            f"Rival              : {(dataframe['archetype']=='Rival').sum()}"
        )

        print(
            f"Strategist         : {(dataframe['archetype']=='Strategist').sum()}"
        )

        print(
            f"Protector          : {(dataframe['archetype']=='Protector').sum()}"
        )

        print(
            f"Mentor             : {(dataframe['archetype']=='Mentor').sum()}"
        )

        print(
            f"Commander          : {(dataframe['archetype']=='Commander').sum()}"
        )

        print(
            f"Anti-Hero          : {(dataframe['archetype']=='Anti-Hero').sum()}"
        )

        print("=" * 70)
        print()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("Anime Character Persona Engine")
    logger.info("Phase 5 - Personality Classifier")
    logger.info("=" * 70)

    classifier = PersonalityClassifier()

    try:

        classifier.load_dataset()

        classifier.load_embeddings()

        classifier.validate()

        classifier.predict()

        classifier.save_report()

        classifier.dataset_statistics()

        classifier.performance()

        classifier.summary()

        logger.info("=" * 70)
        logger.info(
            "PERSONALITY CLASSIFICATION COMPLETED SUCCESSFULLY"
        )
        logger.info("=" * 70)

    except Exception as error:

        logger.exception(
            f"Pipeline Failed : {error}"
        )