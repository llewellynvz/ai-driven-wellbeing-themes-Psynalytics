# 🔍 Enterprise-Level AI-Driven Wellbeing Theme Discovery Pipeline

This repository contains an end-to-end NLP pipeline for **classifying and discovering psychological and wellbeing-related themes** in open-text data. It combines **supervised text classification** with **unsupervised topic modeling** to surface both known and emerging themes in large-scale qualitative datasets.

Designed for organizational psychologists, data scientists, and research teams working with survey data, employee feedback, or social media text.

---

## 🧠 What This Project Does

This pipeline performs two core tasks:

1. **Classifies text** into known psychological or wellbeing themes using a fine-tuned transformer model (e.g., `SetFit` or `paraphrase-mpnet-base-v2`)
2. **Discovers new emerging themes** using unsupervised topic modeling with `BERTopic`

The result: A structured CSV report with both **predicted labels** and **new thematic clusters**, along with UMAP-based interactive visualizations.

---

## 🔧 Project Structure

- `train.csv`: Input dataset with text columns (e.g., survey responses)
- `Theme_Analysis_Results/`: Output folder with classification results, discovered topics, and visualizations
- `theme_discovery.py`: Main pipeline script for classification and BERTopic-based discovery

---

## 📦 Dependencies

Install the core dependencies using:

```bash
pip install pandas numpy torch umap-learn hdbscan seaborn plotly matplotlib scikit-learn transformers sentence-transformers bertopic
```

---

## 🚀 How to Run the Pipeline

### Step 1: Prepare Your Input

Your CSV should include open-ended text in columns like:

```python
["LifeDema", "LifeReso", "PersReso", "WellBe"]
```

These are merged into a single `CombinedText` column before classification and clustering.

### Step 2: Update the Config

In `theme_discovery.py`, update:

```python
MODEL_NAME = "your-fine-tuned-model-path"
```

This should point to a transformer model trained to classify psychological themes (e.g., SetFit or HuggingFace format).

### Step 3: Run the Script

```bash
python theme_discovery.py
```

This will:

- Load your CSV and merge text columns
- Classify each entry into known themes
- Apply BERTopic to surface new topics from embeddings
- Generate:
  - `Classified_Text.csv`
  - `Discovered_Themes.csv`
  - `Theme_Distribution.png`
  - `UMAP_Clusters.html`

---

## 📊 Output Files

After execution, the following will be saved in `Theme_Analysis_Results/`:

- `Classified_Text.csv`: Original data + predicted labels
- `Discovered_Themes.csv`: BERTopic themes and topic names
- `Theme_Distribution.png`: Bar chart of discovered themes
- `UMAP_Clusters.html`: 2D interactive visualization of thematic clusters
- `theme_analysis.log`: Full logs and progress

---

## 🧠 Model Details

The classification pipeline is compatible with any HuggingFace-style model. Recommended:
- `SetFit` models fine-tuned on wellbeing subcategories
- `paraphrase-mpnet-base-v2` for sentence embeddings
- Custom `transformers`-based pipelines

The BERTopic model uses `UMAP` + `HDBSCAN` clustering on sentence embeddings to discover new, unlabeled topics.

---

## 👥 Use Cases

- Classifying and discovering wellbeing concerns in survey or interview data
- Mining employee feedback for recurring stressors and resources
- Surfacing new trends in public discourse (e.g., Reddit, Quora, Twitter)
- Creating dashboards or taxonomies for psychological constructs

---

## 📌 Notes

- This pipeline assumes you have a **fine-tuned classification model** ready
- Text is merged from multiple input columns into a unified "document"
- Unlabeled or low-confidence responses are explored via **unsupervised learning**
- Outputs are saved in user-friendly formats (CSV, PNG, HTML)

---

## 👤 Author

This project was developed by [Prof. Llewellyn van Zyl (PhD)](https://www.linkedin.com/in/llewellynvz), an organizational psychologist and data scientist specializing in wellbeing-focused AI and human-centered analytics.

---

## 📬 Contact

💬 **[Open an Issue](https://github.com/llewellynvz/roberta_training_wellbeing)** on GitHub  
✉️ **Learn more at** [psynalytics.com](https://www.psynalytics.com)

---

Feel free to fork, adapt, or collaborate to evolve this into a modular, production-grade wellbeing insights engine.
