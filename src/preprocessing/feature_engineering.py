"""
==========================================================
Project : Anime Character Persona & Sentiment Engine
Module  : feature_engineering.py
Phase   : Phase 1 - Feature Engineering

Description
-----------
Creates machine learning features from cleaned dialogues.
==========================================================
"""

import logging
import re
from datetime import datetime

import pandas as pd

from config.paths import (
    CLEAN_TEXT_CSV,
    FEATURE_ENGINEERED_CSV
)

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


class FeatureEngineer:

    def __init__(self, input_csv):

        self.input_csv = input_csv

    # -----------------------------------------------------

    def load_data(self):

        logging.info("Loading cleaned dialogue dataset...")

        dataframe = pd.read_csv(self.input_csv)

        logging.info(f"Rows Loaded : {len(dataframe)}")

        return dataframe

    # -----------------------------------------------------
    # Time Utilities
    # -----------------------------------------------------

    @staticmethod
    def time_to_seconds(time_string):

        """
        Converts

        00:00:04,500

        into

        4.5 seconds
        """

        dt = datetime.strptime(
            time_string,
            "%H:%M:%S,%f"
        )

        total_seconds = (

            dt.hour * 3600 +

            dt.minute * 60 +

            dt.second +

            dt.microsecond / 1_000_000

        )

        return total_seconds

    # -----------------------------------------------------

    def subtitle_duration(self, row):

        start = self.time_to_seconds(row["start_time"])

        end = self.time_to_seconds(row["end_time"])

        return round(end - start, 2)

    # -----------------------------------------------------
    # Basic Features
    # -----------------------------------------------------

    @staticmethod
    def sentence_count(text):

        return len(

            re.findall(r"[.!?]", text)

        )

    # -----------------------------------------------------

    @staticmethod
    def average_word_length(text):

        words = text.split()

        if len(words) == 0:

            return 0

        total = sum(

            len(word)

            for word in words

        )

        return round(

            total / len(words),

            2

        )

    # -----------------------------------------------------

    @staticmethod
    def uppercase_count(text):

        return sum(

            1

            for c in text

            if c.isupper()

        )

    # -----------------------------------------------------

    @staticmethod
    def uppercase_ratio(text):

        letters = [

            c

            for c in text

            if c.isalpha()

        ]

        if len(letters) == 0:

            return 0

        upper = [

            c

            for c in letters

            if c.isupper()

        ]

        return round(

            len(upper) / len(letters),

            2

        )
        # -----------------------------------------------------
    # Punctuation Features
    # -----------------------------------------------------

    @staticmethod
    def exclamation_count(text):

        return text.count("!")

    # -----------------------------------------------------

    @staticmethod
    def question_count(text):

        return text.count("?")

    # -----------------------------------------------------

    @staticmethod
    def comma_count(text):

        return text.count(",")

    # -----------------------------------------------------

    @staticmethod
    def ellipsis_count(text):

        return text.count("...")

    # -----------------------------------------------------
    # Dialogue Length Category
    # -----------------------------------------------------

    @staticmethod
    def dialogue_type(word_count):

        if word_count <= 3:

            return "Short"

        elif word_count <= 10:

            return "Medium"

        else:

            return "Long"

    # -----------------------------------------------------
    # Speaking Speed
    # -----------------------------------------------------

    @staticmethod
    def words_per_second(words, duration):

        if duration <= 0:

            return 0

        return round(

            words / duration,

            2

        )

    # -----------------------------------------------------
    # Novel Feature
    # Emotion Intensity Score
    # -----------------------------------------------------

    @staticmethod
    def emotion_intensity(row):

        score = (

            row["exclamation_count"] * 2 +

            row["question_count"] +

            row["ellipsis_count"] +

            row["uppercase_ratio"] * 100

        )

        return round(score, 2)

    # -----------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------

    def engineer_features(self, dataframe):

        logging.info(

            "Creating machine learning features..."

        )

        # Duration

        dataframe["duration_seconds"] = dataframe.apply(

            self.subtitle_duration,

            axis=1

        )

        # Sentence Count

        dataframe["sentence_count"] = dataframe[
            "clean_dialogue"
        ].apply(

            self.sentence_count

        )

        # Average Word Length

        dataframe["average_word_length"] = dataframe[
            "clean_dialogue"
        ].apply(

            self.average_word_length

        )

        # Uppercase Count

        dataframe["uppercase_count"] = dataframe[
            "clean_dialogue"
        ].apply(

            self.uppercase_count

        )

        # Uppercase Ratio

        dataframe["uppercase_ratio"] = dataframe[
            "clean_dialogue"
        ].apply(

            self.uppercase_ratio

        )

        # Exclamation Count

        dataframe["exclamation_count"] = dataframe[
            "clean_dialogue"
        ].apply(

            self.exclamation_count

        )

        # Question Count

        dataframe["question_count"] = dataframe[
            "clean_dialogue"
        ].apply(

            self.question_count

        )

        # Comma Count

        dataframe["comma_count"] = dataframe[
            "clean_dialogue"
        ].apply(

            self.comma_count

        )

        # Ellipsis Count

        dataframe["ellipsis_count"] = dataframe[
            "clean_dialogue"
        ].apply(

            self.ellipsis_count

        )

        # Dialogue Type

        dataframe["dialogue_type"] = dataframe[
            "word_count"
        ].apply(

            self.dialogue_type

        )

        # Words Per Second

        dataframe["words_per_second"] = dataframe.apply(

            lambda row:

            self.words_per_second(

                row["word_count"],

                row["duration_seconds"]

            ),

            axis=1

        )

        # Emotion Intensity

        dataframe["emotion_intensity"] = dataframe.apply(

            self.emotion_intensity,

            axis=1

        )

        logging.info(

            "Feature engineering completed."

        )

        return dataframe
        # -----------------------------------------------------
    # Save Dataset
    # -----------------------------------------------------

    @staticmethod
    def save_data(dataframe):

        dataframe.to_csv(

            FEATURE_ENGINEERED_CSV,

            index=False

        )

        logging.info(

            f"Feature engineered dataset saved:\n{FEATURE_ENGINEERED_CSV}"

        )

    # -----------------------------------------------------
    # Dataset Summary
    # -----------------------------------------------------

    @staticmethod
    def dataset_summary(dataframe):

        print("\n")

        print("=" * 60)

        print("DATASET SUMMARY")

        print("=" * 60)

        print(f"Rows    : {len(dataframe)}")

        print(f"Columns : {len(dataframe.columns)}")

        print("\nColumns:\n")

        for column in dataframe.columns:

            print(f"• {column}")

        print("\n")

        print(dataframe.head())

        print("=" * 60)

        print()


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    logging.info(

        "Starting Feature Engineering Module..."

    )

    engineer = FeatureEngineer(

        CLEAN_TEXT_CSV

    )

    dataframe = engineer.load_data()

    dataframe = engineer.engineer_features(

        dataframe

    )

    engineer.dataset_summary(

        dataframe

    )

    engineer.save_data(

        dataframe

    )

    logging.info(

        "Feature Engineering Completed Successfully."

    )