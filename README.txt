ChargeWise Dashboard | TEB3123 Machine Learning | Group 10
==========================================================

FILES (keep them all in the same folder)
  app.py                  the Streamlit dashboard
  model.joblib            trained Random Forest pipeline
  meta.json               feature lists, dropdown options, fill values
  results.csv             test results for all models and baselines
  tuning.csv              validation results for each model setting
  covid_results.csv       COVID check experiment
  cleaning_log.csv        rows removed at each cleaning step
  feature_importance.csv  permutation importance
  test_predictions.csv    test-set predictions
  clean_data.csv          cleaned dataset for the overview charts
  requirements.txt        exact library versions (needed to load the model)
  .streamlit/config.toml  light theme and colours (hidden folder, keep it!)

RUN LOCALLY (VS Code terminal)
  1. Install Python 3.11 or newer (3.12 or 3.13 recommended).
  2. Open this folder in VS Code, then in the terminal:
       python -m venv venv
       venv\Scripts\activate          (Windows)
       source venv/bin/activate       (Mac / Linux)
       pip install -r requirements.txt
       streamlit run app.py
  3. The dashboard opens at http://localhost:8501

DEPLOY (Streamlit Community Cloud, free)
  1. Create a GitHub repository and upload ALL files in this folder.
     (model.joblib is about 17 MB, under GitHub's 100 MB limit.)
  2. Go to share.streamlit.io, sign in with GitHub, click "Create app".
  3. Pick the repository, branch "main", main file "app.py".
  4. Under Advanced settings, choose Python 3.12 or 3.13, then Deploy.

FONT
  The dashboard uses Apple's system font stack. On Mac and iPhone it shows the real
  SF Pro font; on Windows and Android it shows Inter, the closest free match.
  (SF Pro cannot be embedded on a website under Apple's licence.)

IMPORTANT
  The .streamlit folder starts with a dot, so it may be hidden. Upload it to GitHub too.
  If you re-run the Colab notebook, replace ALL these files (including
  requirements.txt) with the ones from the new dashboard_files.zip.
  Mixing a new model with old version numbers can stop the model loading.
