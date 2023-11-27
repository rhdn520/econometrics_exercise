import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('cps4.csv')
df.head()
df.describe()

df['log_wage'] = np.log(df['wage'])
y = df['log_wage']
x1 = df['educ']
x2 = df['exper']
X = df[['educ','exper']]
X_cons = sm.add_constant(X)


model_1 = sm.OLS(y, X_cons).fit()
model_1.summary()

residual = model_1.resid

# Residual Plot
plt.scatter(x1, residual)
plt.title('Scatter plot of Residuals')
plt.xlabel('Year of Education')
plt.ylabel('Residuals')
plt.axhline(0, color='red', linestyle='--')  # 잔차 0을 기준으로 한 수평선 추가
# plt.show()


model_1 = sm.OLS(y, X_cons).fit(cov_type='HC0')
print(model_1.summary())