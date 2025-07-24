# Employee-Salary-Prediction-ML

🧠 Employee Salary Prediction using Machine Learning

This project predicts an employee's salary based on various features like experience, education level, role, and location using supervised machine learning techniques.


---

📌 Project Overview

The aim of this project is to build a predictive model that can estimate employee salaries using historical data. This model can be useful for HR departments, recruitment agencies, and job seekers.


---

📁 Dataset

The dataset contains the following features:

YearsExperience: Number of years of work experience

EducationLevel: Highest qualification (e.g., Bachelors, Masters, PhD)

JobRole: Role of the employee (e.g., Data Scientist, Software Engineer)

Location: City/Region

Salary: Target variable (Annual Salary in ₹ or $)


You can use your own dataset or a public dataset like:

Kaggle - Salary Prediction Datasets



---

⚙ Technologies Used

Python 🐍

Jupyter Notebook

Pandas

NumPy

Matplotlib / Seaborn (for visualization)

Scikit-learn (for model building)



---

📊 ML Algorithms Used

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Support Vector Regressor (optional)

🧪 Steps Involved

1. Data Cleaning & Preprocessing

Handle missing values

Convert categorical variables using one-hot encoding or label encoding

2. Exploratory Data Analysis

Visualize relationships using scatter plots, boxplots, correlation heatmaps

3. Feature Selection

Select important features that influence salary

4. Model Training

Train multiple models and compare performance using metrics like MAE, MSE, R²

5. Model Evaluation

Evaluate the best model on test data

6. Model Saving

Save the trained model using Pickle

7. Deployment (Optional)

Create a Streamlit web app for real-time salary prediction

📈 Sample Code (Training Model)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X = df[['YearsExperience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

 Example requirements.txt:

pandas
numpy
matplotlib
seaborn
scikit-learn
stremlit
