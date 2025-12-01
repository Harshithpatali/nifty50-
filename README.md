Great — here is a clean, copy-paste \*\*README.md\*\* for your project, explaining the pipeline, API, and how to run everything locally (Airflow + Docker + FastAPI + Streamlit + yfinance).



---



\### 1️⃣ Create the file



```powershell

Set-Location "D:\\Nifty50\_StopPrediction"

New-Item -ItemType File -Path "README.md"

notepad README.md

```



---



\### 2️⃣ Paste this content and save



```md

\# Nifty50 Stop Prediction App (Linear Regression + yfinance + FastAPI + Streamlit + Docker + Airflow)



This is an \*\*end-to-end data analytics \& prediction system\*\* that:



✅ Pulls historical \*\*Nifty 50 Index\*\* data using `yfinance`  

✅ Cleans the dataset to format: `Date, Close, High, Low, Open, Volume`  

✅ Performs \*\*Feature Engineering\*\* (lag, volatility, returns, moving averages, trend signals, date breakdown, volume averages)  

✅ Loads data into \*\*PostgreSQL\*\*  

✅ Trains a \*\*Linear Regression model\*\* and predicts the \*\*next-day Close price\*\*  

✅ Evaluates all regression metrics (R², MAE, MSE, RMSE, MAPE, sMAPE, Bias, Residuals, Durbin-Watson, Max Error, etc.)  

✅ Exposes prediction via a REST API built with \*\*FastAPI\*\*  

✅ Provides a UI using \*\*Streamlit (frontend)\*\*  

✅ Automates the pipeline via \*\*Airflow DAG\*\* to run daily at \*\*6 PM IST\*\*



---



\## 📁 Project Structure



```



Nifty50\_StopPrediction/

│── backend/

│   ├── venv/

│   ├── download\_data.py

│   ├── clean\_data.py

│   ├── feature\_engineering.py

│   ├── load\_to\_postgres.py

│   ├── train\_lr.py

│   ├── evaluate\_lr.py

│   ├── fastapi\_app.py

│   ├── Dockerfile

│   ├── requirements.txt

│   └── .dockerignore

│

│── frontend/

│   ├── streamlit\_app.py

│   ├── Dockerfile

│   └── requirements.txt

│

│── data/

│   ├── raw/

│   ├── clean/

│   └── features/

│

│── mlflow/

│   ├── models/

│   └── artifacts/

│

│── postgres\_data/ (docker volume)

│── .gitignore

└── README.md



````



---



\## 🚀 Local Setup \& Run



\### 1. Create Project Folder (Windows)



```powershell

New-Item -ItemType Directory -Path "D:\\Nifty50\_StopPrediction"

````



\### 2. Set up Backend Environment



```powershell

cd D:\\Nifty50\_StopPrediction\\backend

python -m venv venv

.\\venv\\Scripts\\activate

pip install -r requirements.txt

```



\### 3. Run FastAPI Backend



Inside `backend/` venv activated:



```powershell

uvicorn fastapi\_app:app --host 0.0.0.0 --port 8000 --reload

```



Test in browser:



\* `http://127.0.0.1:8000/docs`

\* `http://127.0.0.1:8000/health`

\* `GET /predict/next-day`



\### 4. Run Streamlit Frontend



In a new terminal:



```powershell

cd D:\\Nifty50\_StopPrediction\\frontend

streamlit run streamlit\_app.py --server.port 8501

```



Open browser:



\* `http://127.0.0.1:8501`



---



\## 🐳 Docker Setup \& Run



From project root:



```powershell

cd D:\\Nifty50\_StopPrediction

docker compose up --build

```



Services:



| Service              | Host Port | Internal Port |

| -------------------- | --------: | ------------: |

| postgres             |      5433 |          5432 |

| backend (FastAPI)    |      8000 |          8000 |

| frontend (Streamlit) |      8501 |          8501 |



---



\## 🔧 Airflow Pipeline Automation



To orchestrate the pipeline, the DAG is stored at:



```

D:\\Nifty50\_StopPrediction\\airflow\\nifty50\_lr\_dag.py

```



\### Run Airflow in WSL (Ubuntu)



```bash

export AIRFLOW\_HOME=~/airflow

mkdir -p $AIRFLOW\_HOME/dags

cp /mnt/d/Nifty50\_StopPrediction/airflow/nifty50\_lr\_dag.py $AIRFLOW\_HOME/dags/

airflow db init

airflow scheduler

airflow webserver -p 8080

```



Then open in Windows browser:



```

http://localhost:8080

```



\* Enable the DAG `nifty50\_lr\_daily\_pipeline`

\* Trigger manually for testing

\* It will run \*\*daily at 6 PM IST\*\*



---



\## 📊 Current Model Performance (Latest Run)



\* Train R²: \*\*0.99945\*\*

\* Test R²: \*\*0.99407\*\*

\* \*\*Predicted next-day Close\*\*: `26122.57`



---



\## 📌 Notes



\* `yfinance` does not need an API key.

\* PostgreSQL creds are stored in `.env` (ignored by git).

\* When using Docker, env vars are injected via `docker compose`.

\* Metrics are calculated in a time-series aware split.



---



\## ✅ Next Improvements (future scope)



\* Save predictions to `nifty50\_predictions` table

\* Add LSTM or advanced ML models

\* CI/CD deployment on cloud (GCP/AWS)

\* Live dashboards \& alerts

\* Model registry via MLflow



---



\## 🧑‍💻 Author



Harshith — Data Analyst / Data Science Engineer



---



Happy building 🚀



```



---



\## 3️⃣ Save and close the editor



✅ `README.md` is now ready.



---



\## Next step?

Choose one, say:



\- `log predictions to db`

\- or `docker check`

\- or `train endpoint improve`



I’ll follow your instructions.

```







\## 🖼 App UI Preview

!\[Nifty50 Stop Prediction UI](assets/ui.png)



