import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print(">>> Script started")

# Load env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Connect to  supported DB using psycopg2
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)

df = pd.read_sql('SELECT * FROM nifty50_features', conn)

print("Rows pulled from DB:", len(df))
print(df.head())

# Use features for prediction — we'll predict Close using numeric features
df = df.sort_values("Date")

# Define X and y
X = df[[
    "Prev_Close", "Return_Sign", "Volatility_5d",
    "MA_5", "MA_20", "HL_Diff", "Days",
    "Day", "Month", "Year", "Weekday", "Volume_MA_5", "Volume"
]]
y = df["Close"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print(">>> Model trained using:", "pandas + scikit-learn")
print("Train R²:", model.score(X_train, y_train))
print("Test R²:", model.score(X_test, y_test))

# ---- Predict next day Close ----
latest_row = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(latest_row)[0]

print("\n📈 **Predicted Next Day Close Price:**", predicted_price)
print(">>> Script finished")

conn.close()
