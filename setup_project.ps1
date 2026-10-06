$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

Write-Host "==> Preparing project environment ..."

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment ..."
    py -3 -m venv .venv
}

Write-Host "Activating virtual environment ..."
. ".\.venv\Scripts\Activate.ps1"

Write-Host "Installing Python dependencies ..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Installing Playwright browser ..."
python -m playwright install chromium

Write-Host ""
Write-Host "Setup complete."

Write-Host ""
Write-Host "Next steps (run from the project root with .venv active):"
Write-Host "  1. Ensure HUGGINGFACEHUB_API_TOKEN is set in your root .env file."
Write-Host "  2. Scrape Philips data: python .\src\step_1_1_philips_scrapper.py"
Write-Host "  3. Scrape Oral-B data: python .\src\step_1_2_oralb_scrapper.py"
Write-Host "  4. Build the retrieval database: python .\src\step2_ingest.py"
Write-Host "  5. Ask questions: python .\src\step3_rag_architecture.py"
Write-Host "  6. Evaluate answers: python .\src\step4_evaluate_rag.py"
Write-Host "Setup does not run these steps automatically."
