"""
==========================================================
Project : Anime Character Persona & Sentiment Engine

Module  : sentiment_analyzer.py

Phase   : Phase 2

Description
-----------
Hybrid Anime Sentiment Analyzer

Uses:
1. VADER Sentiment Analysis
2. Anime Emotion Dictionary
3. Hybrid Sentiment Score

Author : Puviarasan
==========================================================
"""

from __future__ import annotations

import logging

from typing import Dict

import pandas as pd

import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

from config.paths import (

    FINAL_DATASET_CSV,

    SENTIMENT_RESULTS_CSV,

    CHARACTER_SENTIMENT_CSV,

    EMOTION_TIMELINE_CSV

)

from src.nlp.anime_dictionary import (

    ANIME_EMOTION_DICTIONARY

)

# ==========================================================
# Download VADER
# ==========================================================

try:

    nltk.data.find(

        "sentiment/vader_lexicon.zip"

    )

except LookupError:

    nltk.download(

        "vader_lexicon"

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
# Sentiment Analyzer
# ==========================================================

class SentimentAnalyzer:

    """
    Hybrid Anime Sentiment Analyzer
    """

    def __init__(self):

        self.dataframe = None

        self.analyzer = SentimentIntensityAnalyzer()

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        logger.info(

            "Loading multimodal dataset..."

        )

        self.dataframe = pd.read_csv(

            FINAL_DATASET_CSV

        )

        logger.info(

            f"Rows Loaded : {len(self.dataframe)}"

        )

        return self.dataframe

    # ======================================================
    # Anime Dictionary Score
    # ======================================================

    def anime_dictionary_score(

        self,

        dialogue: str

    ) -> float:

        words = str(

            dialogue

        ).lower().split()

        total_score = 0.0

        matched_words = 0

        for word in words:

            word = word.strip(

                ".,!?;:\"'()[]{}"

            )

            if word in ANIME_EMOTION_DICTIONARY:

                total_score += (

                    ANIME_EMOTION_DICTIONARY[

                        word

                    ]

                )

                matched_words += 1

        if matched_words == 0:

            return 0.0

        return round(

            total_score /

            matched_words,

            3

        )

    # ======================================================
    # Hybrid Sentiment
    # ======================================================

    def analyze_dialogue(

        self,

        dialogue: str

    ) -> Dict:

        dialogue = str(dialogue)

        vader = self.analyzer.polarity_scores(

            dialogue

        )

        anime_score = self.anime_dictionary_score(

            dialogue

        )

        hybrid = (

            0.70 *

            vader["compound"]

            +

            0.30 *

            anime_score

        )

        hybrid = max(

            -1,

            min(

                hybrid,

                1

            )

        )

        vader["anime_score"] = round(

            anime_score,

            3

        )

        vader["hybrid_score"] = round(

            hybrid,

            3

        )

        return vader

    # ======================================================
    # Confidence Score
    # ======================================================

    @staticmethod

    def confidence_score(

        score

    ):

        return round(

            abs(score) * 100,

            2

        )

    # ======================================================
    # Sentiment Category
    # ======================================================

    @staticmethod

    def sentiment_category(

        score

    ):

        if score >= 0.50:

            return "Positive"

        elif score <= -0.50:

            return "Negative"

        else:

            return "Neutral"
        # ======================================================
    # Analyze Entire Dataset
    # ======================================================

    def analyze_dataset(self):

        logger.info(
            "Analyzing dialogue sentiments..."
        )

        positive_scores = []
        negative_scores = []
        neutral_scores = []
        compound_scores = []

        anime_scores = []
        hybrid_scores = []

        confidence_scores = []

        sentiment_labels = []

        for dialogue in self.dataframe["clean_dialogue"]:

            scores = self.analyze_dialogue(
                dialogue
            )

            positive_scores.append(
                scores["pos"]
            )

            negative_scores.append(
                scores["neg"]
            )

            neutral_scores.append(
                scores["neu"]
            )

            compound_scores.append(
                scores["compound"]
            )

            anime_scores.append(
                scores["anime_score"]
            )

            hybrid_scores.append(
                scores["hybrid_score"]
            )

            confidence_scores.append(

                self.confidence_score(

                    scores["hybrid_score"]

                )

            )

            sentiment_labels.append(

                self.sentiment_category(

                    scores["hybrid_score"]

                )

            )

        self.dataframe["positive"] = positive_scores

        self.dataframe["negative"] = negative_scores

        self.dataframe["neutral"] = neutral_scores

        self.dataframe["compound"] = compound_scores

        self.dataframe["anime_score"] = anime_scores

        self.dataframe["hybrid_score"] = hybrid_scores

        self.dataframe["confidence"] = confidence_scores

        self.dataframe["sentiment"] = sentiment_labels

        logger.info(
            "Hybrid sentiment analysis completed."
        )

        return self.dataframe

    # ======================================================
    # Most Positive Dialogue
    # ======================================================

    def most_positive_dialogue(self):

        row = self.dataframe.loc[

            self.dataframe["hybrid_score"]

            .idxmax()

        ]

        print("\n")

        print("=" * 70)

        print("MOST POSITIVE DIALOGUE")

        print("=" * 70)

        print(
            f"Character : {row['character']}"
        )

        print(
            f"Dialogue  : {row['dialogue']}"
        )

        print(
            f"Hybrid Score : {row['hybrid_score']}"
        )

        print("=" * 70)

        print()

        return row

    # ======================================================
    # Most Negative Dialogue
    # ======================================================

    def most_negative_dialogue(self):

        row = self.dataframe.loc[

            self.dataframe["hybrid_score"]

            .idxmin()

        ]

        print("\n")

        print("=" * 70)

        print("MOST NEGATIVE DIALOGUE")

        print("=" * 70)

        print(
            f"Character : {row['character']}"
        )

        print(
            f"Dialogue  : {row['dialogue']}"
        )

        print(
            f"Hybrid Score : {row['hybrid_score']}"
        )

        print("=" * 70)

        print()

        return row

    # ======================================================
    # Sentiment Distribution
    # ======================================================

    def emotion_distribution(self):

        distribution = (

            self.dataframe["sentiment"]

            .value_counts()

            .reset_index()

        )

        distribution.columns = [

            "Sentiment",

            "Count"

        ]

        print("\n")

        print("=" * 70)

        print("SENTIMENT DISTRIBUTION")

        print("=" * 70)

        print(distribution)

        print("=" * 70)

        print()

        return distribution
        # ======================================================
    # Character Sentiment Analytics
    # ======================================================

    def character_sentiment(self):

        logger.info(
            "Generating character sentiment analytics..."
        )

        analytics = (

            self.dataframe

            .groupby("character")

            .agg(

                average_positive=("positive", "mean"),

                average_negative=("negative", "mean"),

                average_neutral=("neutral", "mean"),

                average_compound=("compound", "mean"),

                average_anime_score=("anime_score", "mean"),

                average_hybrid_score=("hybrid_score", "mean"),

                average_confidence=("confidence", "mean"),

                total_dialogues=("character", "count")

            )

            .round(3)

            .reset_index()

        )

        print("\n")
        print("=" * 70)
        print("CHARACTER SENTIMENT ANALYTICS")
        print("=" * 70)
        print(analytics)
        print("=" * 70)
        print()

        return analytics

    # ======================================================
    # Emotional Stability Score
    # ======================================================

    def emotional_stability(self):

        logger.info(
            "Calculating emotional stability..."
        )

        stability = (

            self.dataframe

            .groupby("character")["hybrid_score"]

            .std()

            .fillna(0)

            .reset_index()

        )

        stability.columns = [

            "character",

            "hybrid_std"

        ]

        stability["emotional_stability"] = (

            100 -

            (stability["hybrid_std"] * 100)

        )

        stability["emotional_stability"] = (

            stability["emotional_stability"]

            .clip(lower=0)

            .round(2)

        )

        print("\n")
        print("=" * 70)
        print("EMOTIONAL STABILITY")
        print("=" * 70)
        print(stability)
        print("=" * 70)
        print()

        return stability

    # ======================================================
    # Emotion Timeline
    # ======================================================

    def emotion_timeline(self):

        logger.info(
            "Generating emotion timeline..."
        )

        timeline = self.dataframe.copy()

        timeline = timeline.sort_values(
            "subtitle_id"
        )

        timeline["rolling_hybrid_score"] = (

            timeline["hybrid_score"]

            .rolling(

                window=3,

                min_periods=1

            )

            .mean()

            .round(3)

        )

        timeline["emotion_change"] = (

            timeline["hybrid_score"]

            .diff()

            .fillna(0)

            .round(3)

        )

        print("\n")
        print("=" * 70)
        print("EMOTION TIMELINE")
        print("=" * 70)

        print(

            timeline[

                [

                    "subtitle_id",

                    "character",

                    "hybrid_score",

                    "rolling_hybrid_score",

                    "emotion_change"

                ]

            ]

        )

        print("=" * 70)
        print()

        return timeline

    # ======================================================
    # Overall Statistics
    # ======================================================

    def sentiment_statistics(self):

        statistics = {

            "average_positive":

                round(

                    self.dataframe["positive"].mean(),

                    3

                ),

            "average_negative":

                round(

                    self.dataframe["negative"].mean(),

                    3

                ),

            "average_neutral":

                round(

                    self.dataframe["neutral"].mean(),

                    3

                ),

            "average_compound":

                round(

                    self.dataframe["compound"].mean(),

                    3

                ),

            "average_anime_score":

                round(

                    self.dataframe["anime_score"].mean(),

                    3

                ),

            "average_hybrid_score":

                round(

                    self.dataframe["hybrid_score"].mean(),

                    3

                ),

            "average_confidence":

                round(

                    self.dataframe["confidence"].mean(),

                    2

                )

        }

        print("\n")
        print("=" * 70)
        print("HYBRID SENTIMENT STATISTICS")
        print("=" * 70)

        for key, value in statistics.items():

            print(f"{key:30}: {value}")

        print("=" * 70)
        print()

        return statistics
        # ======================================================
    # Save Sentiment Results
    # ======================================================

    def save_sentiment_results(self):

        logger.info(
            "Saving sentiment results..."
        )

        self.dataframe.to_csv(

            SENTIMENT_RESULTS_CSV,

            index=False

        )

        logger.info(

            f"Saved:\n{SENTIMENT_RESULTS_CSV}"

        )

    # ======================================================
    # Save Character Analytics
    # ======================================================

    def save_character_sentiment(self):

        analytics = self.character_sentiment()

        analytics.to_csv(

            CHARACTER_SENTIMENT_CSV,

            index=False

        )

        logger.info(

            f"Saved:\n{CHARACTER_SENTIMENT_CSV}"

        )

    # ======================================================
    # Save Emotion Timeline
    # ======================================================

    def save_emotion_timeline(self):

        timeline = self.emotion_timeline()

        timeline.to_csv(

            EMOTION_TIMELINE_CSV,

            index=False

        )

        logger.info(

            f"Saved:\n{EMOTION_TIMELINE_CSV}"

        )

    # ======================================================
    # Summary
    # ======================================================

    def summary(self):

        print("\n")

        print("=" * 70)

        print("HYBRID ANIME SENTIMENT SUMMARY")

        print("=" * 70)

        print(

            f"Total Dialogues : {len(self.dataframe)}"

        )

        print(

            f"Characters      : {self.dataframe['character'].nunique()}"

        )

        print(

            f"Average Hybrid Score : {round(self.dataframe['hybrid_score'].mean(),3)}"

        )

        print(

            f"Average Confidence : {round(self.dataframe['confidence'].mean(),2)}%"

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
        "Phase 2 - Hybrid Anime Sentiment Engine"
    )

    logger.info("=" * 70)

    analyzer = SentimentAnalyzer()

    try:

        # --------------------------------------------------
        # Load Dataset
        # --------------------------------------------------

        analyzer.load_dataset()

        # --------------------------------------------------
        # Hybrid Sentiment Analysis
        # --------------------------------------------------

        analyzer.analyze_dataset()

        # --------------------------------------------------
        # Analytics
        # --------------------------------------------------

        analyzer.most_positive_dialogue()

        analyzer.most_negative_dialogue()

        analyzer.emotion_distribution()

        analyzer.character_sentiment()

        analyzer.emotional_stability()

        analyzer.emotion_timeline()

        analyzer.sentiment_statistics()

        # --------------------------------------------------
        # Save Results
        # --------------------------------------------------

        analyzer.save_sentiment_results()

        analyzer.save_character_sentiment()

        analyzer.save_emotion_timeline()

        analyzer.summary()

        logger.info("=" * 70)

        logger.info(
            "HYBRID SENTIMENT ENGINE COMPLETED"
        )

        logger.info("=" * 70)

    except Exception as error:

        logger.exception(

            f"Pipeline Failed : {error}"

        )