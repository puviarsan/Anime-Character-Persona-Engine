"""
==========================================================
Project : Anime Character Persona & Sentiment Engine
Module  : subtitle_parser.py
Phase   : Phase 1 - Data Preprocessing

Description
-----------
Reads an SRT subtitle file and converts it into a
structured Pandas DataFrame.

Extracted Fields
----------------
• subtitle_id
• start_time
• end_time
• character
• dialogue
==========================================================
"""

from pathlib import Path
import logging
import re

import pandas as pd

from config.paths import RAW_SUBTITLE_DIR, CLEAN_DIALOGUE_CSV


# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


# ==========================================================
# Subtitle Parser Class
# ==========================================================

class SubtitleParser:
    """
    Reads an SRT subtitle file and converts it into
    a structured Pandas DataFrame.
    """

    def __init__(self, subtitle_path: str | Path):

        self.subtitle_path = Path(subtitle_path)

    # ------------------------------------------------------

    def file_exists(self) -> bool:
        """
        Check whether subtitle file exists.
        """

        if not self.subtitle_path.exists():

            logging.error(
                f"Subtitle file not found:\n{self.subtitle_path}"
            )

            return False

        return True

    # ------------------------------------------------------

    def read_file(self) -> str:
        """
        Read subtitle file.
        """

        logging.info("Reading subtitle file...")

        with open(
            self.subtitle_path,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read()

        logging.info("Subtitle loaded successfully.")

        return content

    # ------------------------------------------------------

    def parse(self) -> pd.DataFrame:
        """
        Parse subtitle file and return DataFrame.
        """

        if not self.file_exists():

            return pd.DataFrame()

        content = self.read_file()

        pattern = re.compile(

            r"(\d+)\s*\n"

            r"(\d{2}:\d{2}:\d{2},\d{3}) --> "

            r"(\d{2}:\d{2}:\d{2},\d{3})\s*\n"

            r"(.*?)(?=\n\s*\n|\Z)",

            re.DOTALL

        )

        matches = pattern.findall(content)

        logging.info(
            f"Total subtitles found : {len(matches)}"
        )

        subtitle_data = []

        for match in matches:

            subtitle_id = int(match[0])

            start_time = match[1]

            end_time = match[2]

            dialogue = match[3].replace("\n", " ").strip()

            character = "Unknown"

            if ":" in dialogue:

                character, dialogue = dialogue.split(":", 1)

                character = character.strip().upper()

                dialogue = dialogue.strip()

            subtitle_data.append(

                {

                    "subtitle_id": subtitle_id,

                    "start_time": start_time,

                    "end_time": end_time,

                    "character": character,

                    "dialogue": dialogue

                }

            )

        dataframe = pd.DataFrame(subtitle_data)

        logging.info(
            "Subtitle parsing completed."
        )

        return dataframe

    # ------------------------------------------------------

    @staticmethod
    def save_csv(
        dataframe: pd.DataFrame,
        output_path: str | Path
    ) -> None:
        """
        Save DataFrame to CSV.
        """

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        logging.info(
            f"CSV saved successfully:\n{output_path}"
        )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    subtitle_file = RAW_SUBTITLE_DIR / "sample.srt"

    parser = SubtitleParser(subtitle_file)

    dataframe = parser.parse()

    print("\nParsed Subtitle Data\n")
    print(dataframe)

    if not dataframe.empty:

        parser.save_csv(

            dataframe,

            CLEAN_DIALOGUE_CSV

        )

        print("\nCSV Saved Successfully!")

    else:

        print("\nNo subtitle data found.")