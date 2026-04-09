import numpy as np
from sklearn.linear_model import LinearRegression
print("\n===== SIMPLE LINEAR REGRESSION =====")
# Example data (1 input feature)
# X = Study hours, Y = Marks
X_simple = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y_simple = np.array([35, 50, 65, 70, 90])
# Create model
model_simple = LinearRegression()
model_simple.fit(X_simple, Y_simple)
# Prediction
prediction_simple = model_simple.predict([[6]])
print("Input (Study Hours):", X_simple.flatten())
print("Actual Output (Marks):", Y_simple)
print("Slope (m):", model_simple.coef_[0])
print("Intercept (c):", model_simple.intercept_)
print("Predicted Marks for 6 hours study:", prediction_simple[0])

print("\n===== MULTIPLE LINEAR REGRESSION =====")
# Example data (2 input features)
# X = [Hours studied, Sleep hours]
# Y = Marks
X_multi = np.array([
    [1, 6],
    [2, 5],
    [3, 6],
    [4, 5],
    [5, 7]
])
Y_multi = np.array([40, 50, 65, 70, 85])
# Create model
model_multi = LinearRegression()
model_multi.fit(X_multi, Y_multi)
# Prediction
prediction_multi = model_multi.predict([[6, 6]])
print("Input (Hours, Sleep):")
print(X_multi)
print("Actual Output (Marks):", Y_multi)
print("Coefficients (m1, m2):", model_multi.coef_)
print("Intercept (c):", model_multi.intercept_)
print("Predicted Marks for [6 hrs study, 6 hrs sleep]:", prediction_multi[0])


# ============================================
# EXPLANATION
# ============================================

"""
EXPLANATION:

1. SIMPLE LINEAR REGRESSION:
- Used when there is only ONE input variable
- Equation: Y = mX + c
    m → slope
    c → intercept
- Example: Marks based on study hours

2. MULTIPLE LINEAR REGRESSION:
- Used when there are MULTIPLE input variables
- Equation: Y = m1X1 + m2X2 + ... + c
- Example: Marks based on study hours + sleep hours

OUTPUT:
- coef_ → slope(s)
- intercept_ → constant value
- predict() → used to predict new values

DIFFERENCE:
- Simple → one input
- Multiple → more than one input
"""
