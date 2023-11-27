import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('cps4.csv')
print(df.head())
print(df.describe())

df['log_wage'] = np.log(df['wage'])

y = df['log_wage']
x1 = df['educ']
x2 = df['exper']
X = df[['educ','exper']]
X_cons = sm.add_constant(X)

model_1 = sm.OLS(y, X_cons).fit()
print(model_1.summary())

model_1_res = model_1.resid
ess_1 = np.sum(model_1_res**2)
print(ess_1)



# Total derivative
x1_cons = sm.add_constant(x1)
model_total = sm.OLS(y, x1_cons).fit()
print(model_total.params[1])

model_aux_3 = sm.OLS(x2, x1_cons).fit()
model_aux_3.summary()
print(model_aux_3.params[1])
a = model_1.params[1] + model_1.params[2] * model_aux_3.params[1]
print(a)


