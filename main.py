import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn import linear_model
import pickle

sns.set()

train_data = pd.read_excel(r"Data_Train.xlsx")
pd.set_option('display.max_columns', None)  # to display all columns
train_data.head()
train_data.info()
# print(train_data['Duration'].value_counts())

train_data["Duration"].value_counts()
train_data.dropna(inplace=True)  # Dropping any row if the data is Null
train_data.isnull().sum()

#######  Expoloratory Data Analysis ########

# clean the Data_of_journey Data like Day and Month
# Clean the Arrival_time Data to get the most suitable data out of it to train our model like Hours or Minutes

# Data Analysis of the Date_of_journey column to get date and month out of it
# After Extracting, we can removing the un-necessary column of Date_of_Journey

train_data['journey_day'] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day
train_data['Journey_month'] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.month
train_data.head()
train_data.drop(['Date_of_Journey'], axis=1, inplace=True)

# Similarly to Date_of_Journey we can extract the values from Dep_time like Hours and Minutes
# After Extracting, we can drop the un-necessary column of Dep_time

train_data["Dep_hour"] = pd.to_datetime(train_data['Dep_Time']).dt.hour
train_data["Dep_min"] = pd.to_datetime(train_data['Dep_Time']).dt.minute
train_data.drop(["Dep_Time"], axis=1, inplace=True)

# Similarly to Date_of_Journey we can extract the values from Arrival_time like Hours and Minutes
# After Extracting, we can drop the un-necessary column of Arrival_time

train_data["Arrival_hour"] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
train_data["Arrival_min"] = pd.to_datetime(train_data['Arrival_Time']).dt.minute
train_data.drop(["Arrival_Time"], axis=1, inplace=True)

# Calculating the Duration (Departure Time - Arrival Time)

duration = list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]

duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extracting the hours data from the duration
    duration_mins.append(
        int(duration[i].split(sep="m")[0].split()[-1]))  # Extracting the Minutes data from the duration

# Adding the Data to the Data Frame

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins
train_data.drop(["Duration"], axis=1, inplace=True)

train_data.info()
#######  Data Handling of the Categorical Data ########
# These data can be Nominal Data [oneHotEncoder] or Ordinal Data [LabelEncoder]

# Like the Airlines in the Data Frame, we will perform oneHotEncoder on this Data
# print(train_data['Airline'].value_counts())

# Visulaize the data to check which Airline has the highest Price
# we use catplot to make a plot between Price and Airline
# According to the data, we can see that the Airline is the Nominal Data.So, we perform oneHotEncoding on it
train_data["Airline"].value_counts()
sns.catplot(y="Price", x="Airline",
            data=train_data.sort_values("Price", ascending=False), kind="boxen", height=6, aspect=3)
# plt.show()

Airline = train_data[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()
print(Airline.head())

# Similarly the Source and Destination is also a Nominal Categorical Data
# According to the data, we can see that the Airline is the Nominal Data.So, we perform oneHotEncoding on it
train_data["Source"].value_counts()
sns.catplot(y="Price", x="Source",
            data=train_data.sort_values("Price", ascending=False), kind="boxen", height=6, aspect=3)
# plt.show()

Source = train_data[["Source"]]
Source = pd.get_dummies(Source, drop_first=True)
Source.head()

train_data["Destination"].value_counts()
sns.catplot(y="Price", x="Destination",
            data=train_data.sort_values("Price", ascending=False), kind="boxen", height=6, aspect=3)
# plt.show()

Destination = train_data[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first=True)
# print(Destination.head())

# The Route and Total_Stops Column contains almost same type of the Data Similarly, the 75%-80% data in the
# Additional_info Column has almost 75%-80% no_info data in it. So, It will not be any effective in out prediction model
# So, we will drop this column from our DataSet

train_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)

# the Total_stops is the Ordinal Categorical Type, So we will perform LabelEncoder
# So, we will assign the individual key Values to the Data

# According to our Data We can observe that Greater the Stops, Greater the Prices
train_data["Total_Stops"].value_counts()
train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

data_train = pd.concat([train_data, Airline, Source, Destination], axis=1)
data_train.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

# Make the Data Analysis on the Test Data
test_data = pd.read_excel(r"Test_set.xlsx")

test_data.dropna(inplace=True)  # Dropping any row if the data is Null
test_data.isnull().sum()

# Date_of_Journey
test_data['journey_day'] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data['Journey_month'] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.month
test_data.drop(['Date_of_Journey'], axis=1, inplace=True)

# Dep_time
test_data["Dep_hour"] = pd.to_datetime(test_data['Dep_Time']).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data['Dep_Time']).dt.minute
test_data.drop(["Dep_Time"], axis=1, inplace=True)

# Arrival Time
test_data["Arrival_hour"] = pd.to_datetime(test_data['Arrival_Time']).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data['Arrival_Time']).dt.minute
test_data.drop(["Arrival_Time"], axis=1, inplace=True)

# Duration
duration = list(test_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]

duration_hours = []
duration_mins = []

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extracting the hours data from the duration
    duration_mins.append(
        int(duration[i].split(sep="m")[0].split()[-1]))  # Extracting the Minutes data from the duration

# Adding the Data to the Data Frame

test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis=1, inplace=True)

# Airline Categorical Data
Airline = pd.get_dummies(test_data["Airline"], drop_first=True)

# Source Categorical Data
Source = pd.get_dummies(test_data['Source'], drop_first=True)

# Destination Categorical Data
Destination = pd.get_dummies(test_data["Destination"], drop_first=True)

# Dropping the Route and Additional_Info Column from the DataSet
test_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)

# Applying LabelEncoder to the Stops Column of the DataSet
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3}, inplace=True)

data_test = pd.concat([test_data, Airline, Source, Destination], axis=1)
data_test.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)
# print("Shape of test Data: ", data_test.shape)

# Feature Selection
# Finding out the best feature which will contribute and have good relation with the target variable
#   1. heatmap
#   2. feature_importance_
#   3. SelectKBest


## Started training the Data

# X = INDEPENDENT FEATURES
X = data_train.loc[:, ['Total_Stops', 'journey_day', 'Journey_month', 'Dep_hour',
                       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
                       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
                       'Airline_Jet Airways', 'Airline_Jet Airways Business',
                       'Airline_Multiple carriers',
                       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
                       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
                       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
                       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
                       'Destination_Kolkata', 'Destination_New Delhi']]

print(X.info())
# y = DEPENDENT FEATURES
y = data_train.iloc[:, 1]
print(y.head())

# Check if is their any correlation between the independent and dependent Attributes

plt.figure(figsize=(18, 18))
sns.heatmap(train_data.corr(), annot=True, cmap="brg_r")

# plt.show()

# Use to give us the important features,
# means if there are two variables that very closely relates, then It is Okay to remove one
# ExtraTreesRegressor help us to find the features importance

selection = ExtraTreesRegressor()
selection.fit(X, y)

plt.figure(figsize=(12, 8))
features_importances = pd.Series(selection.feature_importances_, index=X.columns)
features_importances.nlargest(20).plot(kind='barh')
# plt.show()

# Using RandomForest To fit the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)

y_pred = reg_rf.predict(X_test)
print("<------------ Random Forest Regressor ------------>")
# print("<-------- Accuracy with training Data ------->")
# print(reg_rf.score(X_train, y_train))
#
# print("<-------- Accuracy with test Data ------->")
# print(reg_rf.score(X_test, y_test))

print('<-------------------------------------------------->')
print("Accuracy Using Random Forest Regression : ", end='')
print(reg_rf.score(X_test,y_test))
print('<-------------------------------------------------->')

sns.displot(y_test - y_pred)
# plt.show()

# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel("y_test")
# plt.ylabel("y_pred")
# plt.show()

print('<---------------------- Regression Tree Model ------------------------->')
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Square Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('<----------------------------------------------------------------------->')

print(metrics.r2_score(y_test, y_pred))
# Hyper parameter tuning

# Number of trees in random Forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum Number of levels in Tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

# Minimum Number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# create the random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}

# Random Search of parameters, using 5 fold cross validation,
# search across 100 different Combinations

rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions=random_grid, scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)
prediction = rf_random.predict(X_test)


# Open a file, where you want to store the data about the Model
file = open('flight_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)

model = open('flight_model.pkl', 'rb')
forest = pickle.load(model)

y_prediction = forest.predict(X_test)

print('<-------------------------------------->\n')
print("After tuning the model using RandomizedSearchCV: ", end='')
print(metrics.r2_score(y_test, y_prediction))
print('<-------------------------------------->\n')

x1_train, x1_test, y1_train, y1_test = train_test_split(X, y)
reg_linear = linear_model.LinearRegression()
reg_linear.fit(x1_train, y1_train)

prediction_linear = reg_linear.predict(x1_test)

# plt.scatter(x1_test, prediction_linear, alpha=0.5)
# plt.legend()
# plt.show()

print('<---------------------- linear Regression Model ------------------------->')
print('Mean Absolute Error: ', metrics.mean_absolute_error(y1_test, prediction_linear))
print('Mean Squared Error: ', metrics.mean_squared_error(y1_test, prediction_linear))
print('Root Mean Square Error: ', np.sqrt(metrics.mean_squared_error(y1_test, prediction_linear)))
print('<----------------------------------------------------------------------->')
# sns.displot(y1_test - prediction_linear)
# plt.show()

ans = reg_linear.predict(np.array([[1, 14, 6, 7, 00, 11, 00, 4, 00, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]]))
print(ans)
print('<-------------------------------------->')
print("Accuracy Using Linear Regression: ", end='')
print(reg_linear.score(x1_test, y1_test))
print('<-------------------------------------->')






# plt.plot(x1_train, y1_train, 'o')
# plt.legend()
# m,b = np.polyfit(x1_train, y1_train, 1)
# plt.plot(x1_train, m*x1_train + b)
# plt.show()
print(x1_train.shape)
print(y1_train.shape)
