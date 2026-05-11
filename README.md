# Wuzzuf Job Search Engine 🔍

A Streamlit-based Information Retrieval system for searching job listings in Egypt.

## Features
- **TF-IDF Search**: Fast and relevant search based on job titles.
- **Structured Filters**: Filter by Department, Experience Level, Employment Type, Work Mode, and Governorate.
- **Evaluation Dashboard**: Real-time calculation of Precision, Recall, and F1-score.
- **Batch Evaluation**: Compare system performance across different K values.

## Deployment on Streamlit Cloud
1. Push this repository to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set the main file path to `ir_system_app.py`.
5. Click **Deploy**.

## Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run ir_system_app.py
   ```

## Dataset
The app uses the [Wuzzuf Job Listings Dataset](https://www.kaggle.com/datasets/ahmedhazemelabady/wuzzuf-job-listings-dataset-egypt-january-2025). It automatically downloads it via `kagglehub` if the local CSV is missing.
