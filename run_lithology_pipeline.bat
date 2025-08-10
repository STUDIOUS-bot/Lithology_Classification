@echo off
echo.
echo ========================================
echo 🪨 LITHOLOGY CLASSIFICATION PIPELINE
echo ========================================
echo.

echo 🔧 Installing dependencies...
pip install -r requirements.txt

echo.
echo 🚀 Running ML Pipeline...
python lithology_ml_pipeline.py

echo.
echo ✅ Pipeline completed!
echo 📁 Check model_results/ directory for outputs
echo 🎨 Visualizations saved in model_results/visualizations/
echo.

echo 🌐 To run the web app, use:
echo    streamlit run lithology_streamlit_app.py
echo.

echo 💻 To use CLI, use:
echo    python lithology_cli.py --input your_data.csv
echo.

pause
