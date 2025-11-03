#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Q1

def cumulative_means(sample):
    cumulative = [np.mean(sample[:i + 1]) for i in range(len(sample))]
    return cumulative

alpha = 3
beta = 5
n = 1000  


data = np.random.beta(alpha, beta, n)
cumulative_means_data = cumulative_means(data)
population_mean = alpha / (alpha + beta)


plt.figure(figsize=(10, 6))
plt.plot(range(1, n + 1), cumulative_means_data, label="Cumulative Means")
plt.axhline(y=population_mean, color='r', linestyle='--', label=f"Population Mean ({population_mean:.2f})")
plt.xlabel("Sample Size")
plt.ylabel("Cumulative Mean")
plt.title("Central Limit Theorem")
plt.legend()
plt.grid(True)
plt.show()


# In[6]:


# Q2

def cumulative_means(sample):
    cumulative = [np.mean(sample[:i + 1]) for i in range(len(sample))]
    return cumulative

alpha = 3
beta = 5
n = 1000  
num_replications = 10


replicated_data = [np.random.beta(alpha, beta, n) for _ in range(num_replications)]
cumulative_means_data = [cumulative_means(sample) for sample in replicated_data]
population_mean = alpha / (alpha + beta)

plt.figure(figsize=(10, 6))
for i, cumulative_means_sample in enumerate(cumulative_means_data):
    plt.plot(range(1, n + 1), cumulative_means_sample, label=f"Sample {i + 1}")

plt.axhline(y=population_mean, color='r', linestyle='--', label=f"Population Mean ({population_mean:.2f})")
plt.xlabel("Sample Size")
plt.ylabel("Cumulative Mean")
plt.title("Central Limit Theorem")
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


# Q3

# Part 1
def analyze_data(x):
    x_sorted = sorted(x)
    minimum = x_sorted[0]
    maximum = x_sorted[-1]

    n = len(x)
    q1_index = (n - 1) // 4
    q3_index = 3 * q1_index
    q1 = (x_sorted[q1_index] + x_sorted[q1_index + 1]) / 2
    q3 = (x_sorted[q3_index] + x_sorted[q3_index + 1]) / 2

    median_index = (n - 1) // 2
    median = (x_sorted[median_index] + x_sorted[median_index + 1]) / 2

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [value for value in x_sorted if value < lower_bound or value > upper_bound]

    
    result = {
        'Min': minimum,
        'Q1': q1,
        'M': median,
        'Q3': q3,
        'Max': maximum,
        'IQR': iqr,
        'Outliers': outliers
    }

    return result

# Part 2
x = [2, 36, 12, 14, 204, 21.6, 22.5, 1, 32.8, 32.1, 13, 10, 88, 3.3, 3.1, 88]
result = analyze_data(x)
print(result)


# In[10]:


# Q4

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


diabetes = load_diabetes()


np.random.seed(0)
sample_indices = np.random.choice(diabetes.data.shape[0], 56, replace=False)
sample_X = diabetes.data[sample_indices, :3]  
sample_y = diabetes.target[sample_indices]


def leave_one_out_cross_validation(X, y):
    n = X.shape[0]
    mse = 0
    
    for i in range(n):
        X_i = np.delete(X, i, axis=0)  
        y_i = np.delete(y, i)  
        
        model = LinearRegression().fit(X_i, y_i)  
        y_pred = model.predict(X[i].reshape(1, -1)) 
        
        mse += (y[i] - y_pred) ** 2
    
    rmse = np.sqrt(mse / n)
    return rmse




rmse = leave_one_out_cross_validation(sample_X, sample_y)
formatted_rmse = np.format_float_positional(rmse, precision=2, unique=False, fractional=False)
print(f"Root Mean Squared Error (RMSE): {formatted_rmse}")


# In[11]:


# Q5

def test_statistic(x, y):
    n1 = len(x)
    n2 = len(y)
    
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    s1_squared = np.var(x, ddof=1)
    s2_squared = np.var(y, ddof=1)
    sp_squared = ((n1 - 1) * s1_squared + (n2 - 1) * s2_squared) / (n1 + n2 - 2)
    t = (x_bar - y_bar) / np.sqrt(sp_squared * (1/n1 + 1/n2))
    
    return t


x = np.random.normal(1, 2, 50)
y = np.random.uniform(-2, 2, 57)

observed_t = test_statistic(x, y)
print(f"Observed Test Statistic: {observed_t:.4f}")


# In[ ]:




