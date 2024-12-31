import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#-------------------

df = pd.read_csv(r"file_location\Boston.csv")

#take info about your data

print(f"shape:{df.shape}")

print("#"*50)

print(f"Info:{df.info()}")

print("#"*50)

print(f"decribe:{df.describe()}")

print("#"*50)

#remove unnamed and not important columns

df.drop(df.columns[[14,15,16]],axis=1,inplace=True)

#duplicate rows

print(f"duplicate rows:{df[df.duplicated()]}")

#check missing values

print(f"null values:{df.isnull().sum()}")
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis')

print("#"*50)

#Handling Outlier

fig, ax = plt.subplots(figsize=(50,16)) 
sns.boxplot(data=df, orient="v", palette="Set2")
plt.show


# One column
sns.boxplot(x=df["MEDV"], data=df)
sns.swarmplot(x=df["MEDV"],  data=df, color=".25")



# Remove Outlier

def remove_outlier(col):
    sorted(col)
    Q1, Q3 = col.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    print("Q1 = ", Q1, " Q3 = ", Q3, " IQR = ", IQR)
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


# Numerical data distribution
print(list(set(df.dtypes.tolist())))
df_num = df.select_dtypes(include = ["float64", "int64"])
lst_num_cols = df_num.columns.tolist()
print("Numerical Data : \n",lst_num_cols)
print("\n")


df_copy = df.copy()

indx = 0
for col in lst_num_cols:
    print(indx)
    lower_range, upper_range =  remove_outlier(df_copy[col])
    df_copy[col] = np.where(df_copy[col] < lower_range, lower_range, df_copy[col]) 
    df_copy[col] = np.where(df_copy[col] > upper_range, upper_range, df_copy[col])
    indx = indx +1
    print("-----------------------------")

# no outlier
fig, ax = plt.subplots(figsize=(50,16)) 
sns.boxplot(data=df_copy, orient="v", palette="Set2")
plt.show

#Correlation matrix

# heatmap
corrMatrix = df_copy.corr()

fig, ax = plt.subplots(figsize=(16,16)) 
#sns.heatmap(corrMatrix, annot=True)
sns.heatmap(corrMatrix, annot=True, linewidth=0.01, square=True, cmap="RdBu", linecolor="black")


# Correlation with output variable
cor_target = abs(corrMatrix["MEDV"])

# Selecting highly correlated features : 0.4
relevant_features = cor_target[cor_target>0.4]
print("relevant_features : ",relevant_features.shape,"\n",relevant_features)

print("-----------------------------------------------------------------------")
lst_columns = relevant_features.index.to_list()

my_data = pd.DataFrame(df_copy, columns= lst_columns)
print (my_data.head(10))

print("-----------------------------------------------------------------------")
my_cols = my_data.columns.to_list()
print("List the column names : ",len(my_cols),"\n",my_cols)

X_data = my_data.drop(["MEDV"], axis=1).values
y_data = my_data["MEDV"].values
print("X_data : ",X_data.shape," y_data : ",y_data.shape)

# Standardization
st_scaler = StandardScaler()
st_scaler.fit(X_data)
X_sts = st_scaler.transform(X_data)

# Split into Input and Output Elements

X_train, X_test, y_train, y_test = train_test_split(X_sts, 
              y_data, test_size= 0.20, random_state=100)

print("X_train = ",X_train.shape ," y_train = ", y_train.shape)
print("X_test  = ",X_test.shape ," y_test = ", y_test.shape)

#Linear Regression 
# Training and testing the model
lin_regressor = linear_model.LinearRegression()
lin_regressor.fit(X_train, y_train)

predicted = lin_regressor.predict(X_test)

# Model evaluation
print("Mean Absolute Error    : ", metrics.mean_absolute_error(y_test, predicted))  
print("Mean Squared Error     : ", metrics.mean_squared_error(y_test, predicted))  
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicted)))