import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import train_test_split
from sklearn import tree
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Read the file
df = pd.read_csv("cleveland.csv")
# Delete blank spaces in column names
df.columns = df.columns.str.replace(' ', '')
# Delete the rows where a missing value is found
df_complete_vals = df.loc[(df['MajorVessels'] != '?') 
                       & 
                       (df['thal'] != '?')]
# Split the dataset into X and y (inputs and target)
inputs = df_complete_vals.drop('num', axis='columns')
target = df_complete_vals['num']
# Split cathegorical data into multiple columns in order to have continuous data
# also called binary values. This process is called one hot encoding.
inputs_encoded = pd.get_dummies(inputs, columns=['ChestPainType', 
                                       'RestingElectrocardiographic', 
                                       'ExerciseSlope', 
                                       'thal'])
# Cathegorize values into two main targets: 0 - No heart disease and 1 to 4 - Heart disease
target_not_zero_index = target > 0
target[target_not_zero_index] = 1
# Split the model into train and test
X_train, X_test, y_train, y_test = train_test_split(inputs_encoded, target, test_size=0.2)
# Create the decision tree
# max_depth is used to prevent overfitting
model = tree.DecisionTreeClassifier(min_samples_split=2, max_depth=3)
# Train the model
model.fit(X_train, y_train)
# Get model accuracy
model.score(X_test, y_test)
# Plot the model
fig = plt.figure(figsize=(15, 7.5))
_ = tree.plot_tree(model, 
                   feature_names=inputs_encoded.columns,
                   rounded=True,  
                   class_names=["No Heart Disease", "Heart Disease"],
                   filled=True)
fig.savefig("decistion_tree.png")

# USER INPUT VALUES
while True:
    print("===============================")
    print("Welcome, lets make predictions!")
    print("===============================")
    print("\n\nLet's predict if you are prone to having a heart disease")


    age = int(input("Tell me your age: "))

    sex = int(input("\n0: female\n1: male\nTell me your sex: "))

    print("\n\nideal blood pressure is considered to be between 90/60mmHg and 120/80mmHg")
    print("high blood pressure is considered to be 140/90mmHg or higher")
    print("low blood pressure is considered to be 90/60mmHg or lower\n")
    restingBloodPressure = int(input("Tell me your restingBloodPressure: "))
    
    print("\n\nless than 200 mg/dL for serum cholesterol is OK\n")
    serumCholesterol = int(input("Tell me your serumCholesterol: "))

    print("\n\n0 is bad")
    print("1 is normal\n")
    fastingBloodSugar = int(input("Tell me your fastingBloodSugar: "))
    
    maximumHeartRate = 220 - age

    print("\n\n0 is normal\n1 is anormal exercise induced angina\n")
    exerciseInducedAngina = int(input("Tell me your exerciseInducedAngina: "))
    
    print("\n\n0 to 3.1 is normal STDepression")
    print("3.2 to 6.2 is anormal STDepression\n")
    sTDepression = float(input("Tell me your sTDepression: "))
    
    majorVessels = random.randint(0, 3)

    print("\n\n1 = typical angina")
    print("2 = atypical angina")
    print("3 = non-anginal pain")
    print("4 = asymptomatic\n")
    chestPainType = int(input("Tell me your chest pain type: "))

    if chestPainType == 1:
        chestPainType_1 = 1
        chestPainType_2 = 0
        chestPainType_3 = 0
        chestPainType_4 = 0
    elif chestPainType == 2:
        chestPainType_1 = 0
        chestPainType_2 = 1
        chestPainType_3 = 0
        chestPainType_4 = 0
    elif chestPainType == 3:
        chestPainType_1 = 0
        chestPainType_2 = 0
        chestPainType_3 = 1
        chestPainType_4 = 0
    elif chestPainType == 4:
        chestPainType_1 = 0
        chestPainType_2 = 0
        chestPainType_3 = 0
        chestPainType_4 = 1


    print("\n\n1 = normal")
    print("2 = having ST-T wave abnormality")
    print("3 = showing probable or definite left ventricular hypertrophy\n")
    restingElectrocardiographic = int(input("Tell me your resting electrocardiographic results: "))
    if restingElectrocardiographic == 1:
        restingElectrocardiographic_1 = 1
        restingElectrocardiographic_2 = 0
        restingElectrocardiographic_3 = 0

    elif restingElectrocardiographic == 2:
        restingElectrocardiographic_1 = 0
        restingElectrocardiographic_2 = 1
        restingElectrocardiographic_3 = 0
    elif restingElectrocardiographic == 3:
        restingElectrocardiographic_1 = 0
        restingElectrocardiographic_2 = 0
        restingElectrocardiographic_3 = 1

    print("\n\n1 = upsloping")
    print("2 = flat")
    print("3 = downsloping\n")
    exerciseSlope = int(input("Tell me your exercise slope results: "))
    
    if exerciseSlope == 1:
        exerciseSlope_1 = 1
        exerciseSlope_2 = 0
        exerciseSlope_3 = 0

    elif exerciseSlope == 2:
        exerciseSlope_1 = 0
        exerciseSlope_2 = 1
        exerciseSlope_3 = 0
    elif exerciseSlope == 3:
        exerciseSlope_1 = 0
        exerciseSlope_2 = 0
        exerciseSlope_3 = 1


    print("\n\n3 = normal (no cold spots)")
    print("6 = fixed defect (cold spots during rest and exercise)")
    print("7 = reversible defect (when cold spots only appear during exercise)\n")
    thal = int(input("Tell me your thalium heart scan results: "))

    if thal == 3:
        thal_3 = 1
        thal_6 = 0
        thal_7 = 0

    elif thal == 6:
        thal_3 = 0
        thal_6 = 1
        thal_7 = 0
    elif thal == 7:
        thal_3 = 0
        thal_6 = 0
        thal_7 = 1

    prediction = model.predict([[
    age,
    sex,
    restingBloodPressure,
    serumCholesterol,
    fastingBloodSugar,
    maximumHeartRate,
    exerciseInducedAngina,
    sTDepression,
    majorVessels,
    chestPainType_1,
    chestPainType_2,
    chestPainType_3,
    chestPainType_4,
    restingElectrocardiographic_1,
    restingElectrocardiographic_2,
    restingElectrocardiographic_3,
    exerciseSlope_1,
    exerciseSlope_2,
    exerciseSlope_3,
    thal_3,
    thal_6,
    thal_7
    ]])
    print(prediction[0])
    if prediction[0] is 0:
        print("Hurray!, you are not prone to heart disease :) ")
    elif prediction[0] is 1:
        print("Bad news!, you are about to die!!!")

    acc = float(model.score(X_test, y_test)) * 100
    print("\n Don't worry about the result, this thing is only "  + str(acc)+"% " + "accurate. Go and visit your doctor :)")

    option = int(input("\n\nDo you want to make another prediction? (1 is yes, 0 is no :( )): "))
    if option is 0:
        break