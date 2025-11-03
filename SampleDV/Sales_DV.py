#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Sample sales data
data = {
    "Date": pd.date_range(start="2025-01-01", periods=12, freq='M'),
    "Region": ["North", "South", "East", "West"] * 3,
    "Product": ["A", "B", "C"] * 4,
    "Sales": np.random.randint(1000, 5000, size=12),
    "Profit": np.random.randint(200, 1500, size=12)
}

df = pd.DataFrame(data)
df

#Bar Chart
plt.figure(figsize=(8,5))
sns.barplot(x="Product", y="Sales", data=df, ci=None)
plt.title("Sales by Product")
plt.show()

#Line Chart
plt.figure(figsize=(8,5))
sns.lineplot(x="Date", y="Sales", hue="Region", data=df, marker='o')
plt.title("Sales Over Time by Region")
plt.show()

#Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x="Sales", y="Profit", hue="Product", size="Profit", data=df)
plt.title("Profit vs Sales")
plt.show()

#Pie Chart
sales_by_region = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(6,6))
plt.pie(sales_by_region, labels=sales_by_region.index, autopct="%1.1f%%", startangle=90)
plt.title("Sales Distribution by Region")
plt.show()

#Heatmap
numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(6,4))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Sales and Profit")
plt.show()

#Interactive
fig = px.bar(df, x="Date", y="Sales", color="Region", title="Interactive Sales by Region")
fig.show()


# In[ ]:




