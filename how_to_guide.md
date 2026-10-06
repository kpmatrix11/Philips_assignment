# Philips Sonicare and Oral-B Product RAG

This project scrapes Philips Sonicare and Oral-B electric toothbrush product data, builds a local Chroma retrieval database, answers questions using retrieved product evidence, and evaluates the answers.

## Prerequisites

- Windows 10 or later
- Python 3.11 installed and available through the Python launcher (`py`)
- Internet access for package/model downloads, product scraping, and Hugging Face inference
- A Hugging Face access token with available Inference Provider credits for the RAG assistant and evaluator

Run commands from the project root, the folder containing `requirements.txt`. Use the project virtual environment for every command; do not mix it with another Python installation.

## 0. Download and Open the Project
- Download the project ZIP file.
- Extract all the files into a folder on your computer.
- Open the extracted folder in VS Code.

## 1. Set up the Python environment

Choose **one** of the following options. Both commands assume you are using PowerShell and have Python 3.11 installed.

### Option A: Run the setup script (recommended)

1. Open the project folder in VS Code.
2. Open a PowerShell terminal using **Terminal > New Terminal**. Make sure the terminal is in the project root, the folder containing `setup_project.ps1` and `requirements.txt`. If it is not, change to that folder:

    ```powershell
    Set-Location "C:\path\to\<project folder>>"
    ```
    **project folder** = the folder containing all the file from ZIP.
    Replace the example path with the location where you saved this project.

3. Create a file named "setup_project.ps1" in project folder and paste the contents of "setup_project.txt" (already present in the project folder) into it.

4. Run the setup script:

    ```powershell
    .\setup_project.ps1
    ```

    Step 4 is only needed if while running step3 PowerShell blocks the script.
    
    Try Step 3 first; if you see an execution-policy error, run Step 4 in that same terminal, then step 3 i.e. run setup_project.ps1 again. Otherwise, skip Step 4.

5. If PowerShell says scripts are disabled, allow scripts for this terminal session:

    ```powershell
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    ```



The script creates `.venv` if needed, activates it, installs the required packages and Playwright Chromium, then prints the next commands. It does **not** automatically scrape products, rebuild the database, or run evaluation. When it finishes, continue with Step 2 below in the same terminal.

### Option B: Set up the environment manually

In PowerShell, first change to the project root as shown in Option A. Then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

If PowerShell blocks activation, run this command in the same terminal and activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

After either option, confirm the terminal prompt begins with `(.venv)`. If Python 3.11 is not installed or `py -3.11` fails, install Python 3.11 and reopen PowerShell before continuing.

### Setup check

Before continuing, confirm these setup commands completed successfully (Option A runs them for you):

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```


# The main show starts here: start running the scripts

## Step 1. Scrape product data

Run each scraper from the project root. Both use Playwright Chromium and may take time because they visit product pages.

```powershell
python .\src\step_1_1_philips_scrapper.py
python .\src\step_1_2_oralb_scrapper.py
```

The scrapers save product data under `data\philips\` and `data\oralb\`. Each creates a flat CSV needed by ingestion:

- `data\philips\philips_products_flat.csv`
- `data\oralb\oralb_products_flat.csv`

If these CSVs already exist and you execute this step, it will replace the existing ones.

## 4. Build or refresh the retrieval database

Make sure both flat CSV files above exist, then run:

```powershell
python .\src\step2_ingest.py
```

This downloads/loads the embedding model and creates `chroma_db\`. It removes and rebuilds the existing Chroma database, so run this step whenever you want to refresh the index from the current CSV files.

## 5. Assistant set-up and ask questions

Start the improved interactive RAG assistant:

```powershell
python .\src\step3_rag_architecture_improved.py
```

Enter a question at the `User Query:` prompt. Type `exit` to quit. The assistant prints each response as indented JSON, with product rows under `answer.products`. The first run may download the embedding model. The assistant calls the configured Hugging Face model for query planning and general answers, so a valid token, internet connection, and available inference credits are required. Top-N rating-count questions are ranked directly from numeric product data and do not ask the answer model to choose or mix products.

The original `src\step3_rag_architecture.py` remains available as an unchanged baseline.

**IMPORTANT: If assitant fails to answer your query, the most probable issue is the token limit reached, please try replacing the HUGGINGFACEHUB_API_TOKEN with your key and it should be fine**

## 6. Evaluate the RAG answers

Run The 11-question evaluation from the project root:

```powershell
python .\src\step4_evaluate_rag.py
```

The script prints the question currently being processed and writes the per-question results to `evaluation_results.csv` in the project root. Evaluation makes repeated LLM calls; ensure the Hugging Face account has enough available inference credits.

## Common problems

- **`HUGGINGFACEHUB_API_TOKEN is not set in .env`:** Confirm `.env` is in the project root, the variable name is exact, and the token line has no surrounding quotes.
- **`402 Payment Required`:** The token was accepted, but the account has no remaining Inference Provider credits or access to the configured model. Resolve that with Hugging Face before rerunning model steps.
- **Playwright cannot launch Chromium:** Activate `.venv` and run `python -m playwright install chromium` again.
- **CSV file not found during ingestion:** Run both scraper scripts from the project root, then check the two exact CSV paths listed in Step 3.
- **Missing Python package:** Confirm `(.venv)` is active, then run `python -m pip install -r requirements.txt` again.
- **Wrong working directory or relative path errors:** Change to the project root before running scripts.

## Script order

1. `step_1_1_philips_scrapper.py` and `step_1_2_oralb_scrapper.py` create the product CSVs.
2. `step2_ingest.py` builds the Chroma database from those CSVs.
3. `step3_rag_architecture.py` answers interactive questions using the database.
4. `step4_evaluate_rag.py` evaluates its 11 predefined questions and saves a CSV.

Other files:
- features_considered.csv - feature considered for the current assistant set-up.
- evaluation_results.csv - test results on 11 test cases 
