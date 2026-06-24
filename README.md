# ✈️ Predicting Airline Passenger Satisfaction Using Machine Learning

## 📌 Problem Statement
The airline industry relies heavily on customer satisfaction for loyalty, repeat business, and competitive advantage. This project builds machine learning models to predict whether a passenger is **Satisfied** or **Neutral/Dissatisfied** based on demographic information, travel details, flight delays, and service ratings.

---

## 📂 Project Structure
```
├── Predicting_Airline_Passenger_Satisfaction_Using_Machine_Learning.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset
- **Source:** Custom airline passenger survey dataset (`DS-DATA.csv`)
- **Size:** 129,880 rows × 24 columns
- **Target Variable:** `Satisfaction` → Binary (Satisfied / Neutral or Dissatisfied)

### Features Include:
| Category | Features |
|---|---|
| Demographics | Gender, Age, Customer Type |
| Travel Info | Type of Travel, Class, Flight Distance |
| Service Ratings | Inflight WiFi, Food & Drink, Seat Comfort, Cleanliness, etc. |
| Delay Info | Departure Delay, Arrival Delay |

---

## 🔧 Workflow

### 1. Data Understanding
- Explored shape, datatypes, and distributions
- Identified `ID` as a non-predictive identifier (dropped)
- Fixed datatype issues in `Flight Distance`

### 2. Data Cleaning
- Handled missing values (Arrival Delay: 393 nulls → filled with median)
- Removed duplicates
- Treated outliers (99th percentile cap on Departure Delay)

### 3. Exploratory Data Analysis (EDA)
- Univariate analysis: pie charts for categorical, histograms for numerical
- Bivariate analysis: bar plots of features vs. satisfaction target
- Pearson correlation heatmap → found high correlation between Departure & Arrival Delay → dropped `Arrival Delay`

### 4. Feature Engineering
- One-hot encoding for categorical variables
- Numerical features with <10 unique values treated as categorical
- Feature scaling (StandardScaler) applied for Logistic Regression

### 5. Model Building
Three classifiers were trained and evaluated:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model with StandardScaler |
| Decision Tree | max_depth=12, balanced class weights |
| Random Forest | 200 estimators, max_depth=18, balanced class weights |

### 6. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Cross-validation (5-fold) for generalization check
- Confusion matrix analysis

---

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SmrutiJogin/Predicting-Airline-Passenger-Satisfaction.git
cd Predicting-Airline-Passenger-Satisfaction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Place your `DS-DATA.csv` file in the project root directory and update the file path in the notebook:
```python
df_airline = pd.read_csv('DS-DATA.csv')
```

### 4. Run the notebook
```bash
jupyter notebook Predicting_Airline_Passenger_Satisfaction_Using_Machine_Learning.ipynb
```

---

## 📈 Results Summary

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | ~87% | ~87% |
| Decision Tree | ~94% | ~93% |
| Random Forest | ~97% | ~96% |

> *Random Forest achieved the best performance with highest ROC-AUC score.*

---

## 🙋 Author
**SmrutiJogin**  
[GitHub Profile](https://github.com/SmrutiJogin)
