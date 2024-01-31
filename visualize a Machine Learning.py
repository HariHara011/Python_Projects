import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/Advertising.csv")
print(data.head())
x = data["TV"].values.reshape(-1, 1)
y = data["Sales"]

model = LinearRegression()
model.fit(x, y)  # Corrected line
x_range = np.linspace(x.min(), x.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

import plotly.express as px
import plotly.graph_objects as go
fig = px.scatter(data, x='TV', y='Sales', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, 
                          name='Linear Regression'))
fig.show()
