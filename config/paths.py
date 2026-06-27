"""
==========================================================
Project : Anime Character Persona & Sentiment Engine

Module  : paths.py

Description
-----------
Centralized project paths.

Every module imports paths from here instead of
hardcoding file locations.
==========================================================
"""

from pathlib import Path


# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# Data Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

ANALYTICS_DATA_DIR = DATA_DIR / "analytics"

SAMPLE_DATA_DIR = DATA_DIR / "sample_data"


# ==========================================================
# Raw Data
# ==========================================================

RAW_SUBTITLE_DIR = RAW_DATA_DIR / "subtitles"

RAW_IMAGE_DIR = RAW_DATA_DIR / "images"

RAW_VIDEO_DIR = RAW_DATA_DIR / "videos"


# ==========================================================
# Processed Data
# ==========================================================

PROCESSED_IMAGE_DIR = PROCESSED_DATA_DIR / "processed_images"

CLEAN_DIALOGUE_CSV = PROCESSED_DATA_DIR / "cleaned_dialogues.csv"

CLEAN_TEXT_CSV = PROCESSED_DATA_DIR / "clean_dialogues.csv"

FEATURE_ENGINEERED_CSV = (
    PROCESSED_DATA_DIR /
    "feature_engineered_dialogues.csv"
)

MERGED_DATASET_CSV = PROCESSED_DATA_DIR / "merged_dataset.csv"

# ==========================================================
# Sample Images
# ==========================================================

# ==========================================================
# Sample Image
# ==========================================================

SAMPLE_IMAGE = SAMPLE_DATA_DIR / "sample.jpg"

# ==========================================================
# Processed Images
# ==========================================================

PROCESSED_IMAGE = (
    PROCESSED_IMAGE_DIR /
    "processed_sample.pt"
)

IMAGE_STATISTICS_CSV = (
    PROCESSED_DATA_DIR /
    "image_statistics.csv"
)

IMAGE_QUALITY_REPORT = (
    PROCESSED_DATA_DIR /
    "image_quality_report.json"
)

FINAL_DATASET_CSV = (
    PROCESSED_DATA_DIR /
    "final_multimodal_dataset.csv"
)

DATASET_REPORT = (
    PROCESSED_DATA_DIR /
    "dataset_report.json"
)

# ==========================================================
# NLP Output
# ==========================================================

SENTIMENT_RESULTS_CSV = (
    ANALYTICS_DATA_DIR /
    "sentiment_results.csv"
)

CHARACTER_SENTIMENT_CSV = (
    ANALYTICS_DATA_DIR /
    "character_sentiment.csv"
)

EMOTION_TIMELINE_CSV = (
    ANALYTICS_DATA_DIR /
    "emotion_timeline.csv"
)

# ==========================================================
# Vision Data
# ==========================================================

RAW_IMAGES_DIR = (

    RAW_DATA_DIR /

    "images"

)

IMAGE_EMBEDDINGS_DIR = (

    PROCESSED_DATA_DIR /

    "embeddings"

)

FACE_EMBEDDINGS_FILE = (

    IMAGE_EMBEDDINGS_DIR /

    "face_embeddings.csv"

)

RESNET_FEATURES_FILE = (

    IMAGE_EMBEDDINGS_DIR /

    "resnet_features.csv"

)

# ==========================================================
# Face Detection
# ==========================================================

FACE_OUTPUT_DIR = (

    PROCESSED_DATA_DIR /

    "faces"

)

FACE_REPORT_FILE = (

    ANALYTICS_DATA_DIR /

    "face_detection_report.csv"

)

# ==========================================================
# Models
# ==========================================================

MODEL_DIR = PROJECT_ROOT / "models"

YOLO_FACE_MODEL = MODEL_DIR / "yolov8n-face.pt"


# ==========================================================
# Reports
# ==========================================================

REPORT_DIR = PROJECT_ROOT / "reports"


# ==========================================================
# Dashboard
# ==========================================================

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"


# ==========================================================
# Create folders automatically
# ==========================================================

directories = [

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

    REPORT_DIR,

    DASHBOARD_DIR

]

for directory in directories:

    directory.mkdir(

        parents=True,

        exist_ok=True

    )