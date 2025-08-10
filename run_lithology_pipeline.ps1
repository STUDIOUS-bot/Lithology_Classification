Write-Host ""
Write-Host "========================================" -ForegroundColor Blue
Write-Host "🪨 LITHOLOGY CLASSIFICATION PIPELINE" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

Write-Host "🔧 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "🚀 Running ML Pipeline..." -ForegroundColor Green
python lithology_ml_pipeline.py

Write-Host ""
Write-Host "✅ Pipeline completed!" -ForegroundColor Green
Write-Host "📁 Check model_results/ directory for outputs" -ForegroundColor Cyan
Write-Host "🎨 Visualizations saved in model_results/visualizations/" -ForegroundColor Cyan
Write-Host ""

Write-Host "🌐 To run the web app, use:" -ForegroundColor Magenta
Write-Host "   streamlit run lithology_streamlit_app.py" -ForegroundColor White
Write-Host ""

Write-Host "💻 To use CLI, use:" -ForegroundColor Magenta
Write-Host "   python lithology_cli.py --input your_data.csv" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to continue"
