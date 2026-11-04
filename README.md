# PJM Energy Intelligence – Electricity Demand Forecasting

A Streamlit-based machine learning dashboard that analyzes historical PJM electricity demand and forecasts future hourly demand using an **XGBoost Regressor**.

The application lets users upload hourly electricity-demand data, review historical demand statistics, evaluate model performance, generate forecasts for up to 30 days, and download forecast results as a CSV file.

## Features

- **Dataset upload:** Supports CSV, XLSX, and XLS files.
- **Column selection:** Choose the date/time column and electricity-demand column.
- **Data preparation:** Converts timestamps and demand values, removes invalid rows, sorts observations chronologically, and removes duplicate timestamps.
- **Feature engineering:** Creates calendar features, weekend indicators, lag features, and rolling averages.
- **Model training:** Trains an XGBoost regression model using a chronological 80/20 train-test split.
- **Historical dashboard:** Displays current, average, and peak demand, the latest observation, and the latest seven days of demand.
- **Model evaluation:** Reports MAE, RMSE, and R², with an actual-versus-predicted chart.
- **Future forecasting:** Generates recursive hourly forecasts for a user-selected horizon of 1–30 days.
- **Export:** Displays forecast rows and provides a downloadable CSV file.
- **Feature importance:** Shows the 15 most important model features.

## Project Structure

```text
PROJECT-1_EXCELR/
├── app.py                  # Streamlit dashboard and forecasting workflow
├── PJMW_MW_Hourly.xlsx     # Sample PJM hourly demand dataset
├── xgb_model.pkl           # Serialized XGBoost model artifact
├── feature_columns.pkl     # Saved feature-column metadata
├── holiday_dates.pkl       # Holiday-date metadata
├── metadata.pkl            # Additional model metadata
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

> The current application trains a new XGBoost model from the dataset uploaded in the dashboard. The `.pkl` files are included project artifacts, but the current `app.py` workflow does not load them.

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** XGBoost, scikit-learn
- **Visualization:** Plotly
- **Excel support:** openpyxl

## Dataset

The repository includes `PJMW_MW_Hourly.xlsx`, a sample hourly electricity-demand dataset. Its worksheet is `PJMW_hourly`, with these columns:

| Column | Description |
|---|---|
| `Datetime` | Timestamp of the hourly observation |
| `PJMW_MW` | Electricity demand in megawatts (MW) |

The dashboard also accepts your own CSV or Excel file. It should contain at least:

- One date/time column (for example, `Datetime` or `Timestamp`).
- One numeric electricity-demand column (for example, `PJMW_MW`, `Demand`, or `Load`).
- At least **180 valid observations**. The feature-engineering process uses a 168-hour lag and rolling window.

The app attempts to identify suitable columns automatically, but you can select the correct columns manually after uploading the file.

## Installation

### 1. Clone or download the repository

```bash
git clone <your-repository-url>
cd PROJECT-1_EXCELR
```

If you downloaded the ZIP file, extract it and open a terminal in the `PROJECT-1_EXCELR` folder.

### 2. Create a virtual environment (recommended)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The dashboard uses Plotly for charts. If it is not already listed in your local `requirements.txt`, install it with:

```bash
pip install plotly
```

## Run the Application

From the project directory, execute:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal (typically `http://localhost:8501`). Open that address in your browser.

## How to Use the Dashboard

1. Start the Streamlit application.
2. Upload a CSV, XLSX, or XLS file containing hourly electricity demand.
3. Select the date/time column and demand column if the automatically selected values are not correct.
4. Review the historical overview KPIs and the latest seven days of demand.
5. Check the model evaluation metrics and the actual-versus-predicted chart.
6. Select a forecast horizon from **1 to 30 days** in the sidebar.
7. Click **Generate Forecast**.
8. Review the hourly forecast and use **Download Forecast CSV** to export it.
9. Expand **View model feature importance** to inspect the top features used by the trained model.

## Feature Engineering

The model uses the following input features:

| Feature group | Features |
|---|---|
| Calendar | Hour, day of week, day, month, quarter, year, day of year |
| Weekend | `is_weekend` |
| Demand lags | 1, 2, 3, 24, 48, and 168 hours |
| Rolling statistics | Previous 24-hour and 168-hour rolling mean |

The rolling means are shifted by one observation so the current target value is not included in those historical averages.

## Model Configuration and Evaluation

The application trains an `XGBRegressor` with the following configuration:

- `n_estimators=400`
- `max_depth=7`
- `learning_rate=0.04`
- `subsample=0.85`
- `colsample_bytree=0.9`
- `objective="reg:squarederror"`
- `random_state=42`

The time-ordered feature dataset is split into 80% training data and 20% testing data. The dashboard reports:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted demand.
- **RMSE (Root Mean Squared Error):** Square root of the average squared prediction error; larger errors receive more weight.
- **R² Score:** Indicates how well predictions explain the variation in the test data.

Metric values depend on the uploaded dataset and should be interpreted in the context of its date range and demand characteristics.

## Forecasting Approach

Forecasting is recursive. The model predicts the next hour, adds that prediction to the demand history, recalculates lag and rolling features, and then predicts the following hour. This process repeats for the selected forecast horizon.

Because later predictions use earlier predicted values, forecast uncertainty can increase farther into the future. Results are estimates, not guaranteed electricity demand values.

## Troubleshooting

- **Unable to read the file:** Confirm that the upload is a valid CSV or Excel file and is not corrupted.
- **Date/time parsing issues:** Ensure the selected timestamp column contains recognizable date/time values.
- **Invalid demand values:** Ensure the selected demand column is numeric and measured consistently (MW is expected for the supplied sample).
- **Not enough observations:** Provide at least 180 valid timestamp-demand rows.
- **Missing Plotly module:** Run `pip install plotly`.
- **Streamlit command not found:** Activate the virtual environment and install dependencies, or run `python -m streamlit run app.py`.

## Notes and Limitations

- The application expects hourly observations because its lag features are defined in hours.
- Missing or irregular timestamps are not automatically resampled to a regular hourly frequency.
- The model is trained when a valid dataset is uploaded; it is not loaded from the included serialized model file.
- Forecast accuracy depends on data quality, history length, seasonality, and changes in electricity-demand behavior.
- The dashboard requires an internet connection to load the externally hosted background image and Google font used in its styling; core forecasting runs locally after dependencies are installed.

## Future Improvements

- Add automated handling of missing hours and irregular time intervals.
- Include holiday and weather variables in the forecasting features.
- Compare XGBoost with statistical and deep-learning forecasting models.
- Add cross-validation and prediction intervals.
- Add model persistence and model-loading options.
- Provide downloadable historical evaluation results and forecast visualizations.

