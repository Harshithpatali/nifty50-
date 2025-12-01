import os
import pandas as pd
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import pandas.api.types as ptypes

print(">>> Script started")

# -------- Load env --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
print("BASE_DIR:", BASE_DIR)
print("ENV_PATH:", ENV_PATH)

if os.path.exists(ENV_PATH):
    print("Loading .env...")
    load_dotenv(ENV_PATH)
else:
    raise FileNotFoundError(f".env file not found at {ENV_PATH}")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

print("POSTGRES_DB:", POSTGRES_DB)
print("POSTGRES_USER:", POSTGRES_USER)

if not all([POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD]):
    raise ValueError("Postgres credentials missing in .env")


def get_conn():
    print(
        f"Connecting to Postgres at {POSTGRES_HOST}:{POSTGRES_PORT} / {POSTGRES_DB} as {POSTGRES_USER}"
    )
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    conn.autocommit = False
    return conn


def map_dtype_to_pg(dtype):
    """Map pandas dtype to PostgreSQL type."""
    if ptypes.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    if ptypes.is_integer_dtype(dtype):
        return "BIGINT"
    if ptypes.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if ptypes.is_bool_dtype(dtype):
        return "BOOLEAN"
    return "TEXT"


def create_table_from_df(conn, df: pd.DataFrame, table_name: str):
    cols = []
    for col in df.columns:
        col_type = map_dtype_to_pg(df[col].dtype)
        cols.append(f'"{col}" {col_type}')

    cols_sql = ", ".join(cols)


# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Basic Metrics
r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred_test)/y_test)) * 100
smape = np.mean(2 * np.abs(y_test - y_pred_test)/(np.abs(y_test) + np.abs(y_pred_test))) * 100
evs = explained_variance_score(y_test, y_pred_test)
max_err = max_error(y_test, y_pred_test)
bias = np.mean(y_pred_test - y_test)
resid_std = np.std(y_test - y_pred_test)

# Adjusted R²
n = len(y_test)
p = X_train.shape[1]
adj_r2 = 1 - (1-r2) * (n-1)/(n-p-1)

# Durbin-Watson
dw = sm.stats.durbin_watson(y_test - y_pred_test)

# Coeff table
coeffs = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_
})

print("\n📊 Evaluation Metrics:")
print("R²:", r2)
print("Adjusted R²:", adj_r2)
print("Explained Variance Score:", evs)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape, "%")
print("sMAPE:", smape, "%")
print("Bias:", bias)
print("Residual Std Dev:", resid_std)
print("Max Error:", max_err)
print("Durbin-Watson:", dw)
print("\n🔍 Coefficients:")
print(coeffs)
print("\n🧠 Feature Impact Table:")
print(coeffs)