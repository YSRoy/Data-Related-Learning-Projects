#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

pd.set_option('display.max_columns', None)

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


# In[13]:


# basic stats
df.describe()


# In[14]:


# Total Sales and Profit by Region
sales_by_region = df.groupby("Region")[["Sales","Profit"]].sum()
sales_by_region


# In[15]:


# Average Sales by Product
avg_sales_product = df.groupby("Product")["Sales"].mean()
avg_sales_product


# In[16]:


# Top performing product
df.sort_values(by="Sales", ascending=False)


# In[18]:


# Sales above 4000
high_sales = df[df["Sales"] > 4000]
high_sales


# In[20]:


# Profit margin calculation
df["Profit_Margin"] = df["Profit"] / df["Sales"] * 100
df


# In[22]:


#Bar Chart
plt.figure(figsize=(8,5))
sns.barplot(x="Product", y="Sales", data=df, ci=None)
plt.title("Sales by Product")
plt.show()


# In[23]:


#Line Chart
plt.figure(figsize=(8,5))
sns.lineplot(x="Date", y="Sales", hue="Region", data=df, marker='o')
plt.title("Sales Over Time by Region")
plt.show()


# In[24]:


#Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x="Sales", y="Profit", hue="Product", size="Profit", data=df)
plt.title("Profit vs Sales")
plt.show()


# In[25]:


#Pie Chart
sales_by_region_sum = df.groupby("Region")["Sales"].sum()
plt.figure(figsize=(6,6))
plt.pie(sales_by_region_sum, labels=sales_by_region_sum.index, autopct="%1.1f%%", startangle=90)
plt.title("Sales Distribution by Region")
plt.show()


# In[26]:


#Heatmap
numeric_df = df.select_dtypes(include='number')
plt.figure(figsize=(6,4))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Sales, Profit, and Profit Margin")
plt.show()


# In[27]:


#Interactive
fig = px.bar(df, x="Date", y="Sales", color="Region", title="Interactive Sales by Region")
fig.show()


# In[28]:


#Trends
# Monthly sales trend
monthly_sales = df.groupby("Date")["Sales"].sum()
monthly_sales.plot(figsize=(8,5), marker='o', title="Total Monthly Sales")
plt.ylabel("Sales")
plt.show()


# In[ ]:




