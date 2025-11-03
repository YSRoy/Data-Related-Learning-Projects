#!/usr/bin/env python
# coding: utf-8

# # Py Project

# ## Introduction
# 
# This project is about exploring factors affecting liver disease.

# ### Dataset Description
# The dataset contains information related to hepatitis, with 155 instances and 20 attributes, including the class attribute. The data was donated by G.Gong from Carnegie-Mellon University via Bojan Cestnik of Jozef Stefan Institute.

# ### Variable Types and Statistical Measures
# 
# Categorical Variables: Type of variables that represent categories or groups. These variables can take on a limited, fixed number of distinct values or labels, and there is no inherent order or numerical significance among these categories. 
# Class (DIE, LIVE), SEX (male, female), STEROID (no, yes), ANTIVIRALS (no, yes), FATIGUE (no, yes), MALAISE (no, yes), ANOREXIA (no, yes), LIVER BIG (no, yes), LIVER FIRM (no, yes), SPLEEN PALPABLE (no, yes), SPIDERS (no, yes), ASCITES (no, yes), VARICES (no, yes), HISTOLOGY (no, yes)
# 
# Ordinal Variables: Type of categorical variable that, in addition to having distinct categories, also have a meaningful order or ranking among them. However, the intervals between the categories are not necessarily uniform or measurable.
# AGE (10, 20, 30, 40, 50, 60, 70, 80)
# 
# Continuous Variables: Quantitative variables that can take on an infinite number of values within a given range. These variables are typically measured on a continuous scale and can include decimal values.
# BILIRUBIN, ALK PHOSPHATE, SGOT, ALBUMIN,PROTIME
# 
# Graphical Displays and Statistical Measures:
# 
# Categorical Variables:
# Bar charts to show the distribution of each category.
# Class distribution can be visualized using a bar chart.
# 
# Ordinal Variables:
# Histogram to show the distribution of ages.
# 
# Continuous Variables:
# Box plots to identify outliers.
# Histograms for a visual representation of continuous variable distributions.

# ### Dataset:

# In[6]:


import pandas as pd
df = pd.read_csv("hepatitis.txt")

print(df.info())
print(df.describe())


# ### Data Cleaning:
# 
# Data cleaning plays a pivotal role in the data preparation phase as it entails the identification and rectification of errors or inconsistencies within a dataset. This essential process guarantees the accuracy, comprehensiveness, and analysis readiness of the data. Typical data cleaning activities involve addressing missing values, eliminating duplicates, rectifying data types, and getting rid of irrelevant or redundant information. The primary objective is to improve the dataset's quality, alleviate potential biases or inaccuracies, and establish a more dependable basis for meaningful analysis and interpretation.

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ["Class", "AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA",
                "LIVER BIG", "LIVER FIRM", "SPLEEN PALPABLE", "SPIDERS", "ASCITES", "VARICES",
                "BILIRUBIN", "ALK PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]

missing_values = {"BILIRUBIN": ["?"], "ALK PHOSPHATE": ["?"], "SGOT": ["?"], "ALBUMIN": ["?"], "PROTIME": ["?"]}

df = pd.read_csv("hepatitis.txt", names=column_names, na_values=missing_values)
df.dropna(inplace=True)
df["BILIRUBIN"] = df["BILIRUBIN"].astype(float)

# Save the cleaned dataset
df.to_csv("hepatitis_cleaned.csv", index=False)
print(df.info())
print(df.describe())


# ## Data Analysis:

# #### Hypthesis Testing

# In[8]:


import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

df = pd.read_csv("hepatitis_cleaned.csv")
selected_factors = ['AGE', 'SEX', 'ANTIVIRALS', 'BILIRUBIN', 'SGOT', 'HISTOLOGY']


class_1_data = df[df['Class'] == 1]
class_2_data = df[df['Class'] == 2]


continuous_columns = ['BILIRUBIN', 'SGOT'] 

for column in continuous_columns:
    t_statistic, p_value = ttest_ind(class_1_data[column].dropna(), class_2_data[column].dropna())
    print(f'Test for {column}:')
    print(f'T-statistic: {t_statistic}')
    print(f'P-value: {p_value}')
    if p_value < 0.05:
        print('Reject the null hypothesis. There is a significant difference.\n')
    else:
        print('Fail to reject the null hypothesis. There is no significant difference.\n')


categorical_columns = ['SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']

# Perform chi-square tests for categorical variables
for column in categorical_columns:
    contingency_table = pd.crosstab(df['Class'], df[column])
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    print(f'Test for {column}:')
    print(f'Chi-square statistic: {chi2_stat}')
    print(f'P-value: {p_value}')
    if p_value < 0.05:
        print('Reject the null hypothesis. There is a significant difference.\n')
    else:
        print('Fail to reject the null hypothesis. There is no significant difference.\n')


# In summary, the tests indicate significant differences in several variables(BILIRUBIN, MALAISE, LIVER BIG, LIVER FIRM, SPLEEN PALPABLE, SPIDERS, ASCITES, VARICES, HISTOLOGY), suggesting meaningful associations or patterns in the dataset for those specific factors. Other factors (SGOT, SEX, STEROID, ANTIVIRALS, FATIGUE, ANOREXIA) do not have any significant differnece.

# #### Correlation Analysis

# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("hepatitis_cleaned.csv")
selected_columns = ['BILIRUBIN', 'MALAISE', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']
selected_data = df[selected_columns]

correlation_matrix = selected_data.corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# BILIRUBIN and MALAISE have a very weak negative correlation (-0.034376).
# BILIRUBIN and HISTOLOGY have a moderate positive correlation (0.268529).
# MALAISE and HISTOLOGY have a weak positive correlation (0.132362).

# #### Model

# In[14]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("hepatitis_cleaned.csv")
df = df.dropna()

selected_features = ['BILIRUBIN', 'MALAISE', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']
X = df[selected_features]
y = df['Class']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print("Confusion Matrix:")
print(conf_matrix)

# 8. Feature Importance
n_top_features = 5 

feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
sorted_importance = feature_importance.sort_values(ascending=False).head(n_top_features)
print(sorted_importance)

plt.figure(figsize=(10, 6))
sorted_importance.plot(kind='bar', rot=45) 
plt.title('Top {} Feature Importance'.format(n_top_features))
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()


# The model's accuracy is decent at 81.2%, but other metrics should be considered, especially if there's an imbalance in the dataset.
# The precision is relatively low, indicating that when the model predicts positive, it has a high chance of being incorrect.
# The recall is moderate, suggesting that the model is reasonably good at capturing actual positive instances.
# The F1 score balances precision and recall, and a score of 0.4 suggests room for improvement.

# HISTOLOGY and BILIRUBIN are the most important features for predicting the outcome, contributing significantly to the model's decision-making process.
# VARICES_1, LIVER FIRM_2, and MALAISE also contribute to the model's predictions, albeit to a lesser extent.

# ## Conclusion:
# Based on the analyses, it is evident that certain clinical and laboratory features, such as BILIRUBIN levels, histological conditions, and the presence of specific symptoms (e.g., MALAISE), play crucial roles in predicting the outcome. The model, while showing decent accuracy, requires further refinement, especially in reducing false positives and improving recall.

# In[ ]:




