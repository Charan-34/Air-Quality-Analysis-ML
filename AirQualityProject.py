import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
df = pd.read_csv("city_day.csv")
#EDA
print(df.info())
print(df.describe())
##Objective 1: Data Cleaning and Handling Missing Values

##Aim:
##To identify and handle missing values using Pandas.
# Check missing values
print(df.isnull().sum())

# Fill missing numeric values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)
# Cleaning duplicate values from dataset
df.drop_duplicates(inplace=True)
# Filter AQI greater than 200
high_aqi = df[df['AQI'] > 200]

# Sort by AQI
sorted_df = df.sort_values(by='AQI', ascending=False)

print(high_aqi.head())

# Scatter Plot → relationship between PM2.5 and AQI
sns.scatterplot(x='PM2.5', y='AQI', data=df)
plt.title("PM2.5 vs AQI")
plt.show()

# Bar Plot → average AQI by city
df.groupby('City')['AQI'].mean().head(10).plot(kind='bar')
plt.title("Average AQI by City")
plt.show()


#Objective 2
##Compare City-wise AQI using Boxplot

##Aim:
##To visualize AQI variation across different cities.

top_cities = df['City'].value_counts().head(5).index
sns.boxplot(data=df[df['City'].isin(top_cities)], x='City', y='AQI')
plt.title("City-wise AQI Comparison")
plt.show()

##Objective 3 
#detecting and fixing outliers


#skewness
plt.hist(df['AQI'].dropna(), bins=30)
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.title("Histogram of AQI (Skewness Analysis)")
plt.show()
#outlier detect using IQR
Q1 = df['AQI'].quantile(0.25)
Q3 = df['AQI'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['AQI'] < lower_bound) | (df['AQI'] > upper_bound)]

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
#before fixing
outliers_before = df[(df['AQI'] < lower_bound) | (df['AQI'] > upper_bound)]
print("Outliers before capping:", len(outliers_before))

#after fixing
df.loc[df['AQI'] < lower_bound, 'AQI'] = lower_bound
df.loc[df['AQI'] > upper_bound, 'AQI'] = upper_bound
outliers_after = df[(df['AQI'] < lower_bound) | (df['AQI'] > upper_bound)]
print("Outliers after capping:", len(outliers_after))

#verify
plt.hist(df['AQI'], bins=30)
plt.title("AQI after Outlier Capping")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()

#OBjective 4
##Aim:
##To understand the role of Machine Learning by predicting AQI using Linear Regression.
X = df[['PM2.5', 'PM10', 'NO2', 'CO']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))



##Objective 5: Analyze Correlation using Heatmap
##
##Aim:
##To study relationships between AQI and pollutants.

corr = df[['AQI', 'PM2.5', 'PM10', 'NO2', 'CO']].corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()


#T test

# Extract drop-na AQI data for both cities
delhi_aqi = df[(df['City'] == 'Delhi') & (df['AQI'].notnull())]['AQI']
mumbai_aqi = df[(df['City'] == 'Mumbai') & (df['AQI'].notnull())]['AQI']

t_stat, p_value = stats.ttest_ind(delhi_aqi, mumbai_aqi, equal_var=False)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4e}")

alpha = 0.05
if p_value < alpha:
    print("Reject the Null Hypothesis: There is a statistically significant difference in the mean AQI between Delhi and Mumbai.")
else:
    print("Fail to reject the Null Hypothesis: There is no significant difference in the mean AQI.")
# Z-test
sample = df['AQI'].dropna()
z = (sample.mean()) / (sample.std()/np.sqrt(len(sample)))
print("Z value:", z)
