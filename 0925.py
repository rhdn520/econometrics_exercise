import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("APT_data.csv", encoding = 'cp949')


df.rename(columns={'단지규모':'block', '시세':'price', '연령':'age', '평수':'sqft'}, inplace=True)

# print(df.head())
# print(df.tail())
# df.info()


y = df['price']
x1 = df['age']
# x2 = df['block']

# x1_cons = sm.add_constant(x1)
# model_1 = sm.OLS(y, x1_cons).fit()
# print(model_1.summary())


x_mat = df[['age','sqft']]
x_mat_cons = sm.add_constant(x_mat)
model_m = sm.OLS(y, x_mat_cons).fit()
print(model_m.summary())
