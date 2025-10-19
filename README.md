# Stock-Price-Prediction
A deep learning project to forecast stock prices using an LSTM neural network. 🛠️ Built with Python, TensorFlow, Keras, yfinance, and Pandas.
# Stock Price Prediction with LSTM 📈

![Project Status](https://img.shields.io/badge/status-complete-green)
![Tech Stack](https://img.shields.io/badge/tech-Python%20%7C%20TensorFlow%20%7C%20Pandas-blue)

This project forecasts the closing price of Apple (AAPL) stock using a Long Short-Term Memory (LSTM) deep learning model. It demonstrates a complete workflow from data acquisition and time-series preprocessing to model training and evaluation.

---

## 🎯 Problem Statement

Stock market prediction is a famously difficult problem. The goal is to predict the future price of a financial asset, but these prices are influenced by countless factors (news, company performance, global events) and are notoriously "noisy" and non-linear.

A simple statistical model (like Linear Regression) fails because it cannot capture the **temporal dependencies** (i.e., the price today *depends* on the prices from the last 60 days).

This project solves this challenge by implementing a **Long Short-Term Memory (LSTM) network**, a specialized type of Recurrent Neural Network (RNN) designed to:

1.  **Handle Time-Series Data:** LSTMs are built to recognize patterns in sequences of data, making them perfect for stock charts.
2.  **Feature Engineering:** It moves beyond simple prices by using a 60-day "lookback" period as a feature, allowing the model to learn from historical trends.
3.  **Advanced Modeling:** It provides a far more accurate prediction than basic models by "remembering" important patterns from the past and "forgetting" irrelevant noise.

---

## 🛠️ Tech Stack & Tools

* 🐍 **Language:** Python 3
* 🧠 **Deep Learning:** TensorFlow & Keras (Sequential API, LSTM, Dense, Dropout layers)
* 📦 **Data Handling:** Pandas & NumPy
* 📊 **Data Visualization:** Matplotlib & Seaborn
* 📈 **Data Source:** `yfinance` (Yahoo Finance API)
* ⚙️ **Preprocessing:** `scikit-learn` (MinMaxScaler)
* 📓 **Environment:** Google Colab / Jupyter Notebook

---

## 🚀 Project Pipeline (Methodology)

The project follows a standard data science pipeline for time-series forecasting:

1.  **Data Acquisition:** Fetched 10+ years of daily stock data for Apple (AAPL) using the `yfinance` library.
2.  **Data Preprocessing:**
    * Filtered the dataset to use only the 'Close' price.
    * Scaled the data to a range of (0, 1) using `MinMaxScaler` to help the neural network converge faster.
3.  **Feature Engineering (Sequencing):**
    * Transformed the linear list of prices into sequences.
    * A **60-day "lookback" period** was used. The prices from `Day 1` to `Day 60` (features, $X$) are used to predict the price on `Day 61` (label, $y$).
4.  **Train/Test Split:** Split the data into 80% for training and 20% for testing.
5.  **Model Architecture:**
    * Built a stacked `Sequential` LSTM model in Keras.
    * **Layer 1:** `LSTM(64 units)` with `return_sequences=True` (to feed the next layer).
    * **Layer 2:** `LSTM(64 units)` (last LSTM layer).
    * `Dropout(0.2)` layers were added to prevent overfitting.
    * **Output Layer:** `Dense(1 unit)` to predict the single 'Close' price.
6.  **Model Training:**
    * Compiled the model using the **'adam'** optimizer and **'mean_squared_error'** as the loss function.
    * Trained the model for 25 epochs.
7.  **Evaluation:**
    * Generated predictions on the unseen test data.
    * Performed an **inverse transform** on the scaled predictions to get the actual dollar values.
    * Calculated the **Root Mean Squared Error (RMSE)** to quantify the model's accuracy.

---

## 📊 Results & Evaluation

The primary evaluation metric is the **Root Mean Squared Error (RMSE)**, which measures the average difference between the model's predictions and the actual stock prices.

The final plot visually confirms the model's high accuracy. The **'Predicted Price'** (green line) line closely tracks the **'Actual Price'** (orange line) in the test dataset, demonstrating that the model successfully learned the complex patterns of the stock's movement.



---

## 🏃 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/[YOUR-REPO-NAME].git
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy yfinance matplotlib seaborn
    ```
3.  **Run the notebook:**
    Open the `.ipynb` file in Jupyter Notebook or Google Colab and run all the cells sequentially.



