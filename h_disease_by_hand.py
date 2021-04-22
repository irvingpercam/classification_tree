import numpy as np 
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas.core.common import SettingWithCopyWarning
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

totalinputs = inputs_encoded
totalinputs['num'] = target

class Node():
    # Constructor
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, information_gain=None, value=None):
        
        # Decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.information_gain = information_gain
        
        # Leaf node
        self.value = value

class DecisionTreeClassifier():
    # Constructor
    def __init__(self, min_samples_split=2, max_depth=2):
        
        # Tree Root Initialization
        self.root = None
        
        # Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # Split until the conditions are not satisfied
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:

            # Get the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)

            # If the information gain is positive
            if best_split["information_gain"]>0:

                # Left recursion
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)

                # Right recursion
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)

                # Return the decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["information_gain"])
        
        # Computing leaf node
        leaf_value = self.calculate_leaf_value(Y)

        # Return the leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        
        # Dictionary that stores the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # loop over all the feature values present in the data
            for threshold in possible_thresholds:

                # Get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                # Check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                    # Compute Information Gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")

                    # Update best split
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["information_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # Return Best Split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if int(feature_val)<=int(tree.threshold):
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)



X = totalinputs.iloc[:, :-1].values
Y = totalinputs.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

model = DecisionTreeClassifier(min_samples_split=2, max_depth=3)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test) 

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

    acc = float(accuracy_score(Y_test, Y_pred)) * 100
    print("\n Don't worry about the result, this thing is only "  + str(acc)+"% " + "accurate. Go and visit your doctor :)")

    option = int(input("\n\nDo you want to make another prediction? (1 is yes, 0 is no :( )): "))
    if option is 0:
        break