# Data Science 3 Project     Group No.: 44


#Importing Libraries : 
    
# from sklearn.decomposition import PCA
import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Dictionary to store the maximum accuracy for all algorithms
D={}



warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# Reading the csv file
df=pd.read_csv('parkinsons.csv')

#Extracting ATTributes of Dataset : 
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values



# Performing Min - Max Normalisation On DataSet (Except Class Attribute):
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)

y=labels

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)

# *******************************************************************************************

#Function for Classification using Support Vector Machine Algorithm (SVM) :
def SVM_Algo ():
    
    print("*******************************************************************")
    print()
    print("      Classification Using SVM Algorithm      ")
    print("___________________________________________________________________")
    print()
    
    
    #Splitting The dataset into training data and test data : 
    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)
    
    #fitting the model in SVM
    classifi2 = SVC(kernel='linear')
    classifi2.fit(x_train,y_train)
    
    y2_pred = classifi2.predict(x_test)
     #Finding Accuracy Score for SVM Classification
    Acc_Svm=accuracy_score(y_test, y2_pred)*100
    
    #Plotting confusion Matrix : 
    cm=confusion_matrix(y_test,y2_pred)
    sn.heatmap(cm, annot=True,cmap="Blues")  
    plt.title("Confusion Matrix for SVM Classifier")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.show()
    
    #
    D["SVM"]=round(Acc_Svm,3)
    print("Accuracy Score : ", round(Acc_Svm,3)," %")
    print()
    print()
    print()
    

# KNN Algorithm

# function for calculating euclidean distance
def Euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2            #(x1-x2)**2+(y1-y2)**2
    return distance**0.5

# function to classify the class for a test data
def get_class(x_train,y_train,test_data,k):
    pred_class = []
    for i in range(len(x_train)):
        dist = Euclidean_distance(x_train[i],test_data) #calculating euclidean distance
        temp = []
        temp.append(dist)
        temp.append(y_train[i])
        pred_class.append(temp)
    pred_class.sort()
    count_1 = 0
    count_0 = 0

    for i in range(k):
        if(pred_class[i][1]==1):
            count_1 += 1
        else:
            count_0 += 1

    if(count_1>count_0):
        return 1
    return 0

# Function for calculating the accuracy
def classify(x_train,x_test,y_train,y_test,k):
    test_pred = []
    for i in range(len(x_test)):
        c = get_class(x_train,y_train,x_train[i],k)
        test_pred.append(c)
    return test_pred


def KNN_Algo():
    print("*******************************************************************")
    print("*******************************************************************")
    print()
    print("      Classification Using KNN Classifier Algorithm      ")
    print("____________________________________________________________________")
    print()
    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)
    k = [1,5,9,13]
    accuracy = []
    for i in k:
        pred = classify(x_train,x_test,y_train,y_test,i)
        count = 0
        for j in range(len(pred)):
            if(pred[j]==y_test[j]):
                count += 1
        acc = (count/len(pred))*100
        cm=confusion_matrix(y_test,pred)
        sn.heatmap(cm, annot=True,cmap="Blues")  
        plt.title("Confusion Matrix for k = "+str(i)+" ,KNN")
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.show() 
        print("Accuracy for k = "+str(i)+" is "+str(acc) )
        print()
        accuracy.append(acc)
    D["KNN"]=round(max(accuracy))
    print("maximum accuracy for knn is" ,max(accuracy))
    plt.bar(k,accuracy)
    plt.xticks(k)
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for KNN algorithm")
    plt.show()
    print()
    print()

        
def Mod_KNN_Algo ():
    print("*******************************************************************")
    print("*******************************************************************")
    print()
    print("      Classification Using Modified KNN Classifier Algorithm      ")
    print("____________________________________________________________________")
    print()
    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)
    k = [1,3,5,7]
    # pca = PCA(n_components = 10)
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)
    D2={}

    #confusion matrix for k = 1,3,5,7
    k=[1,3,5,7]
    acc=[]
    for i in k:
        print("For k = ",i," : ")
        # classifier that consider k-nearest neighbours to determine class
        model = KNeighborsClassifier(n_neighbors=i,
                weights = 'distance',
                metric = 'mahalanobis',
                metric_params={'VI': np.cov(np.linalg.inv(np.cov(x_train.T)), rowvar=False)})         
        model.fit(x_train,y_train)                      # .fit() use regression and genrate a model (used on training data)
        y_pred = model.predict(x_test)                  # .predict() is used on test data to determine its class(label)
        cm = confusion_matrix(y_test, y_pred)           # creaitng a confusion matrix
        # print("The confusion matrix for normalized data for k=",i, " is ", cm)
        plt.figure(figsize=(7,5))
        sn.heatmap(cm, annot=True,cmap="Blues")         # plot rectangular data as color coded matrix
        # plt.xlabel('Predicted')
        # plt.ylabel('Truth')
        plt.title("Confusion Matrix for k = "+str(i)+" ,Modified KNN")
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        a1 = accuracy_score(y_test, y_pred)             # compute subset accuracy
        acc.append(a1)
        plt.show()
        print("The accuracy score for normalized data for k=",i ," is  ",round(a1*100,3)," %")
        D2[i]=round(a1*100,3)
        print()
    print()

    plt.bar(k,acc)
    plt.xlabel('K values')
    plt.xticks(k)
    plt.ylabel('Accuracy')
    plt.title('Modified KNN classifier')
    plt.show()
    k2 = max(D2, key = D2.get)
    m=D2[k2]
    print("Accuracy score is maximum corresponding to k = ",k2," And it's value is : ",m)
    D["Modified KNN"]=m
    
    print()
    print()
    print()
    
        
def RF_Algo () :
    
    print("*******************************************************************")
    print("*******************************************************************")
    print()
    print("      Classification Using Random Forest Classifier Algorithm      ")
    print("____________________________________________________________________")
    print()
    print()
    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=42)
    #Random Forest algorithm
    # Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for bthe the classification and Regression proglems in ML. 
    #it is based on the concept of ensembling learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model

    #implementing the algorithm
    #Fitting the Decision Tree classifier to the training set
    classifier = RandomForestClassifier(n_estimators= 40, max_depth = 15, random_state = 50)
    #n_estimators is the required no of trees in the Random Forest. The default value is 10. we can choose any number but need to take care of the overfitting issue
    #max_depth parameter specifies the maximum depth of each tree. The default values is None., which means each tree will expand until every leaf is pure. A pure leaf is one where all of the data on the leaf comes from the same class
    #the accuracy of the model gets saturated i.e even on changing the n_estimators and max_depth value the accuracy of the model remains nearly the same


    #fitiing the model on the training data
    classifier.fit(x_train, y_train)

    #predicting the values from the test data
    Y_predict = classifier.predict(x_test)

    #printing the Accuracy score
    accuracy_random = accuracy_score(y_test, Y_predict) 
    print("Accuracy :", round(accuracy_random*100,3)," %")

    #printing the confusion matrix
    confusion_random = confusion_matrix(y_test, Y_predict)
    # print("the confusion matrix is:", confusion_matrix(y_test, Y_predict))
    #confusion matrix is of 2 X 2 because it has two classes

    #plotting the heat map to visualise the confusion matrix
    # hm = sns.heatmap(c)
    sn.heatmap(confusion_random/np.sum(confusion_random), annot=True, cmap='Blues') #this will show the percentage of the data
    plt.ylabel("predicted label")
    plt.xlabel("True label")
    plt.title("Confusion Matrix for RandomForest")
    plt.show()
    
    D["Random Forest"] = round(accuracy_random*100,3)
    print()
    print()
    print()

def GMM1():
    print("*******************************************************************")
    print("*******************************************************************")
    print()
    print(" Classification Using Unimodal Gaussian Mixture Model Classifier   ")
    print("____________________________________________________________________")
    print()
    print()
    # print("Unimodal Gaussian Distribution ")
    # Reading csv using pandas

    df=pd.read_csv("parkinsons.csv")
    df=df.drop('name',axis=1)
    #Manking copy of dataframe df
    X = df.copy()
    #Status column of dataframe
    X_label = df['status']

    #Spliting the given data into training and test data
    [X_train, X_test, X_label_train, X_label_test] = train_test_split(X, X_label, test_size=0.3, random_state=42, shuffle=True)

    df3=X_train     
    df4=X_test
    # Creating the dataframes for different status
    df_0=df3.loc[df3["status"]==0,:]        
    df_1=df3.loc[df3["status"]==1,:]       
    # Deleting status column from the testing and training data before proceeding for bayes classification

    df4=df4.drop('status',axis=1)
    df_0=df_0.drop('status',axis=1)
    df_1=df_1.drop('status',axis=1) 
    #Mean of columns and Covariance Matrix of training data for status 0 and 1
    mean_0=df_0.mean()     
    cov_0=df_0.cov()      
    mean_1=df_1.mean()    
    cov_1=df_1.cov()      

    #List for storing predicted status of test samples   
    k=[]        
    # print(df4)
    for i in range(df4.shape[0]):
        p=np.matmul(np.transpose(df4.iloc[i]-mean_0),np.linalg.inv(cov_0))
        CCD_0=(np.exp(-(np.matmul(p,df4.iloc[i]-mean_0))/2))/((2*np.pi)*(np.linalg.det(cov_0)*0.5))#Likelihood of status 0
        q=np.matmul(np.transpose(df4.iloc[i]-mean_1),np.linalg.inv(cov_1))
        CCD_1=(np.exp(-(np.matmul(q,df4.iloc[i]-mean_1))/2))/((2*np.pi)*(np.linalg.det(cov_1)*0.5))#Likelihood of status 1

     # prior probability for classes

        PC_0=len(df_0)/len(df3)      
        PC_1=len(df_1)/len(df3)      
        P_0=(CCD_0*PC_0)/(CCD_1*PC_1+CCD_0*PC_0)
        P_1=(CCD_1*PC_1)/(CCD_1*PC_1+CCD_0*PC_0)
        if (P_0>P_1):
            k.append(0)
        else:
            k.append(1)

    #Confusion Matrix
    conf_mat=confusion_matrix(X_label_test.to_numpy(),np.array(k))
    sn.heatmap(conf_mat, annot=True,cmap="Blues") 
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title("Confusion Matrix for Unimodel Gaussian")
    
  

    #Classification  Accuracy
    accu_score=accuracy_score(X_label_test.to_numpy(),np.array(k)) 
    D["Unimodal Gaussian"]=round(accu_score*100,3)    
    print('Accuracy :',round(accu_score*100,3),"%")
    print()
    print()
    print()
    
    
def GMM2():
    
    print("*******************************************************************")
    print("*******************************************************************")
    print()
    print(" Classification Using  Gaussian Mixture Model   ")
    print("____________________________________________________________________")
    print()
    print()
   
    df=pd.read_csv("parkinsons.csv")
    df=df.drop('name',axis=1)
    #Manking copy of dataframe df
    X = df.copy()
    #status column of dataframe
    X_label = df['status']

    #Spliting the given data into training and test data
    [X_train, X_test, X_label_train, X_label_test] = train_test_split(X, X_label, test_size=0.3, random_state=42, shuffle=True)

    train_data = X_train
    test_data = X_test
    test_data2 = test_data.pop('status')

    # Building Separate status for GMM

    train_status_1 = train_data[train_data['status'] == 1]
    train_status_1.pop('status')

    train_status_0 = train_data[train_data['status'] == 0]
    train_status_0.pop('status')


    # Computing the prior probablity 
    p0 = train_status_0.shape[0]/(train_status_0.shape[0]+train_status_1.shape[0])
    p1 = train_status_1.shape[0]/(train_status_0.shape[0]+train_status_1.shape[0])
    #Using for loop for predicting status for different values of Q using GMM
    
    li = [3,5,7,9,11]
    acc_sc=[]
    D3={}
    for q in li:
        print("For q = ",q," : ")
        # Fitting the model for status0 and status1 for Q=q
        GMM0 = GaussianMixture(
            n_components=q, covariance_type='full', random_state=42, reg_covar=1e-4).fit(train_status_0)
        GMM1 = GaussianMixture(
            n_components=q, covariance_type='full', random_state=42, reg_covar=1e-4).fit(train_status_1)
        # List to store the predicted status 
        res = [] 
        for a,test_sample in test_data.iterrows():  # Iterating over each test sample
            # Computing loglikelihood probablity
            logliklehood_0 = GMM0.score_samples([test_sample])[0]
            logliklehood_1 = GMM1.score_samples([test_sample])[0]
            # Computing likelihood of sample in each status
            liklehood_0 = np.exp(logliklehood_0)
            liklehood_1 = np.exp(logliklehood_1)

            # If P(X/C0)*P(C0)> P(X/C1)*P(C1) means sample is of 0 status as per Bayes classifier
            if((p0*liklehood_0)/(p0*liklehood_0+p1*liklehood_1))>((p1*liklehood_1)/(p0*liklehood_0+p1*liklehood_1)):
                res.append(0)
            else:
                 res.append(1)
        #Finding Accuracy
        acc_score = ( res == test_data2).sum() /  (test_data2.shape[0])  
        acc_sc.append(round(100*acc_score,3))
        

        
        #The Confusion Matrix
        mat = confusion_matrix(test_data2,  res)
        sn.heatmap(mat, annot=True,cmap="Blues") 
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.title("Confusion Matrix for Q = "+str(q)+" ,GMM")
        plt.plot()
        plt.show()
        
        D3[q]=acc_score
        print("Accuracy Score : ", round(acc_score*100,3)," %" )
        print()
    #Plotting bar graph of accuracy for different values of Q
    plt.bar(li,acc_sc)
    plt.title("GMM: Accuracy for different values of Q")
    plt.xlabel("Q values")
    plt.xticks(li)
    plt.ylabel("Accuracy")
    plt.show()
    k3 = max(D3, key = D3.get)
    m=D3[k3]
    print("Accuracy score is maximum corresponding to Q = ",k3," And it's value is : ",round(m*100,3)," %")
    D["GMM"]=round(m*100,3)
    plt.show()
    
   
SVM_Algo()  
KNN_Algo() 
Mod_KNN_Algo() 
RF_Algo () 
GMM1()
GMM2()


method = list(D.keys())
values = list(D.values())
  
# # fig = plt.figure(figsize = (10, 5))
 
# # creating the bar plot

plt.bar(method, values, color ='Red', width = 0.4) 
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.title("Accuracy for Different Algorithms")
plt.show()

print()

P=pd.DataFrame(D.items(), columns=['Algorithm', 'Accuracy'])
print(P)


    
        


