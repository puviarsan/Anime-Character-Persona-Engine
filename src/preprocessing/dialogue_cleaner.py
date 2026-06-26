"""
==========================================================
Project : Anime Character Persona & Sentiment Engine
Module  : dialogue_cleaner.py
Phase   : Phase 1 - Data Preprocessing

Description
-----------
This module cleans subtitle dialogues and creates
additional features for NLP.

Output Features
---------------
• clean_dialogue
• word_count
• character_count
==========================================================
"""

import logging
import re

import pandas as pd

from config.paths import (
    CLEAN_DIALOGUE_CSV,
    CLEAN_TEXT_CSV
)

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


# ---------------------------------------------------------
# Dialogue Cleaner
# ---------------------------------------------------------

class DialogueCleaner:

    def __init__(self, input_csv):

        self.input_csv = input_csv

    # -----------------------------------------------------

    def load_data(self):

        logging.info("Loading dialogue dataset...")

        dataframe = pd.read_csv(self.input_csv)

        logging.info(f"Rows Loaded : {len(dataframe)}")

        return dataframe

    # -----------------------------------------------------

    @staticmethod
    def clean_text(text):

        if pd.isna(text):

            return ""

        text = str(text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing spaces
        text = text.strip()

        return text

    # -----------------------------------------------------

    def clean_dataframe(self, dataframe):

        logging.info("Cleaning dialogue...")

        dataframe["clean_dialogue"] = dataframe["dialogue"].apply(
            self.clean_text
        )

        # Character Count
        dataframe["character_count"] = dataframe[
            "clean_dialogue"
        ].str.len()

        # Word Count
        dataframe["word_count"] = dataframe[
            "clean_dialogue"
        ].str.split().str.len()

        # Remove empty dialogues
        dataframe = dataframe[
            dataframe["clean_dialogue"] != ""
        ]

        # Remove duplicate rows
        dataframe = dataframe.drop_duplicates()

        dataframe.reset_index(
            drop=True,
            inplace=True
        )

        logging.info("Dialogue cleaning completed.")

        return dataframe

    # -----------------------------------------------------

    @staticmethod
    def save_data(dataframe):

        dataframe.to_csv(

            CLEAN_TEXT_CSV,

            index=False

        )

        logging.info(

            f"Clean dataset saved:\n{CLEAN_TEXT_CSV}"

        )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    cleaner = DialogueCleaner(

        CLEAN_DIALOGUE_CSV

    )

    df = cleaner.load_data()

    cleaned_df = cleaner.clean_dataframe(df)

    print("\n")

    print(cleaned_df)

    print("\n")

    cleaner.save_data(cleaned_df)