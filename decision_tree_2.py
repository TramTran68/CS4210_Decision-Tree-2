#-------------------------------------------------------------------------
# AUTHOR: Tram Tran
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train, test decision trees, and output the accuracy of each model created by 
# using each of 3 training sets on the test set
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
    spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
    astigmatism_map = {'Yes': 1, 'No': 2}
    tear_map = {'Reduced': 1, 'Normal': 2}

    for data in dbTraining:
        X.append([age_map[data[0]], spectacle_map[data[1]], astigmatism_map[data[2]], tear_map[data[3]]])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    class_map = {'Yes': 1, 'No': 2}

    for data in dbTraining:
        Y.append(class_map[data[4]])

    #Total accuracy of the model over 10 runs
    total_accuracy = 0.0

    #Loop your training and test tasks 10 times here
    for i in range (10):

        #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        dbTest = []
        X_test = []

        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                   dbTest.append(row)

        #Count the total number of correct predictions
        correct_predictions = 0
        total = len(dbTest)

        for data in dbTest:
            #Transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            X_test = [[
                age_map[data[0]],
                spectacle_map[data[1]],
                astigmatism_map[data[2]],
                tear_map[data[3]]
            ]]
            true_label = class_map[data[4]]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            predicted_label = clf.predict(X_test)[0]
            if predicted_label == true_label:
                correct_predictions += 1
            
        #Accumulate the total accuracy after each run
        if total > 0:
            total_accuracy += correct_predictions / total

    #Find the average of this model during the 10 runs (training and test set)
    average_accuracy = total_accuracy / 10

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f'Final accuracy when training on {ds}: {average_accuracy:.2f}')




