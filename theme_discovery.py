################################################################################
# ENTERPRISE-LEVEL AI THEME DISCOVERY PIPELINE
# - Uses a fine-tuned transformer model (SetFit + paraphrase-mpnet-base-v2)
# - Classifies text into known themes
# - Uses unsupervised learning to find new emerging themes
# - Generates CSV reports & visualizations
################################################################################

import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from bertopic import BERTopic
from logging.handlers import RotatingFileHandler

################################################################################
# 1. CONFIGURATION
################################################################################

# Set paths
DATA_FILE = "train.csv"  # Your input dataset
OUTPUT_FOLDER = "Theme_Analysis_Results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set model name (fine-tuned version)
MODEL_NAME = "your-fine-tuned-model-path"  # Update with the correct path

# Columns containing text
TEXT_COLUMNS = ["LifeDema", "LifeReso", "PersReso", "WellBe"]

# Logging configuration
LOG_FILE = os.path.join(OUTPUT_FOLDER, "theme_analysis.log")
rotating_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
rotating_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), rotating_handler]
)
logger = logging.getLogger(__name__)


################################################################################
# 2. LOAD DATA
################################################################################

def load_and_merge_data(file_path, text_cols):
    """
    Load CSV, merge multiple text columns into a single 'CombinedText' column.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    df = pd.read_csv(file_path, dtype=str, encoding="utf-8", on_bad_lines="skip")
    df.fillna("", inplace=True)

    df["CombinedText"] = df[text_cols].apply(lambda row: " ".join(row.values.astype(str)).strip(), axis=1)

    # Drop empty rows after merging
    df = df[df["CombinedText"].str.strip() != ""].reset_index(drop=True)

    logger.info(f"Loaded dataset with {df.shape[0]} rows.")
    return df


################################################################################
# 3. APPLY CLASSIFIER
################################################################################

def classify_text(df, model_path):
    """
    Classify text using a fine-tuned transformer model.
    """
    logger.info("Loading fine-tuned transformer model...")
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

    logger.info("Classifying text...")
    df["PredictedLabel"] = df["CombinedText"].apply(lambda text: classifier(text, truncation=True)[0]['label'])

    output_path = os.path.join(OUTPUT_FOLDER, "Classified_Text.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Classification results saved to {output_path}")

    return df


################################################################################
# 4. DISCOVER NEW THEMES (UNSUPERVISED)
################################################################################

def discover_new_themes(df):
    """
    Use BERTopic to find new themes from misclassified or uncertain text.
    """
    logger.info("Extracting text embeddings...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    embeddings = embedding_model.encode(df["CombinedText"].tolist(), show_progress_bar=True)

    logger.info("Applying BERTopic...")
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
    topics, _ = topic_model.fit_transform(df["CombinedText"].tolist(), embeddings)

    df["DiscoveredTopic"] = topics
    df["TopicName"] = df["DiscoveredTopic"].apply(lambda x: topic_model.get_topic(x)[0][0] if x != -1 else "Outlier")

    output_path = os.path.join(OUTPUT_FOLDER, "Discovered_Themes.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"New theme discovery results saved to {output_path}")

    return df, topic_model


################################################################################
# 5. VISUALIZATION
################################################################################

def plot_theme_distribution(df):
    """
    Plot the frequency of discovered themes.
    """
    theme_counts = df["TopicName"].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=theme_counts.index, y=theme_counts.values, palette="viridis")
    plt.xticks(rotation=90)
    plt.title("Theme Distribution")
    plt.ylabel("Count")
    plt.xlabel("Theme Name")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "Theme_Distribution.png"))
    plt.close()
    logger.info("Theme distribution plot saved.")


def plot_umap_clusters(df, embeddings):
    """
    Use UMAP to visualize clusters in 2D space.
    """
    logger.info("Generating UMAP visualization...")
    reducer = umap.UMAP(n_components=2, metric="cosine")
    umap_embeddings = reducer.fit_transform(embeddings)

    df["UMAP_X"] = umap_embeddings[:, 0]
    df["UMAP_Y"] = umap_embeddings[:, 1]

    fig = px.scatter(df, x="UMAP_X", y="UMAP_Y", color="TopicName", title="UMAP Clustering of Discovered Themes")
    fig.write_html(os.path.join(OUTPUT_FOLDER, "UMAP_Clusters.html"))
    logger.info("UMAP plot saved as HTML.")


################################################################################
# 6. EXECUTE PIPELINE
################################################################################

def main():
    logger.info("=== STARTING THEME DISCOVERY PIPELINE ===")

    df = load_and_merge_data(DATA_FILE, TEXT_COLUMNS)
    df = classify_text(df, MODEL_NAME)
    df, topic_model = discover_new_themes(df)

    plot_theme_distribution(df)
    plot_umap_clusters(df, topic_model.get_document_info()["embedding"])

    logger.info("=== THEME DISCOVERY PIPELINE COMPLETED ===")


if __name__ == "__main__":
    main()
