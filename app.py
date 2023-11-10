import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from ydata_profiling import ProfileReport
with st.sidebar:
    st.image("ml.jpeg")
    st.header("AutoML App")
    choice=st.radio("Data OPER",["Uploading","Preprocessing","ML Model","Prediction"])
if choice=="Uploading":
    file=st.file_uploader("Upload you data")
    if file is not None:
        data=pd.read_csv(file)
        st.dataframe(data)
        data.to_csv("sourcefile.csv")
if choice=="Preprocessing":
    if os.path.exists("sourcefile.csv"):
        st.info("Action cannot be back if you want to back upload again")
        data=pd.read_csv("sourcefile.csv")
        if st.button("GET DATA(Also Apply Filter)"):
            st.dataframe(data)
        if st.button("GET INFO"):
            print(data.info())
            st.dataframe(data.describe())
        dropC=st.selectbox("Choose Column you want to drop",data.columns)
        if st.button("DROP COLUMN"):
            data.drop([dropC],axis=1,inplace=True)
            data.to_csv("sourcefile.csv",index=False)
        if st.button("Get Detailed Report"):
            report=ProfileReport(data,title="AutoML Report")
            report.to_file("report.html")
        if st.button("Handle Missing and Duplicated"):
            valueFill=st.number_input("Fill Value")
            if st.button("Fill N/A"):
                data=data.fillna(valueFill)
            if st.button("DROP N/A"):
                data=data.dropna()
            st.info("Duplicates- {}".format(data.duplicated().sum()))
            duplicateSC=st.selectbox("Subset",data.columns)
            if st.button("Drop Duplicates by Sub Column"):
                data=data.drop_duplicates(duplicateSC)
            if st.button("Drop Duplicates"):
                data=data.drop_duplicates()
            data.to_csv("sourcefile.csv",index=False)
        if st.button("ONE HOT ENCODING"):
            data=pd.get_dummies(data,drop_first=True)
            st.dataframe(data)
            data.to_csv("sourcefile.csv")
    else:
        st.error("Firstly upload your file")
if choice=="ML Model":
    if os.path.exists("sourcefile.csv"):
        st.header("Select your target and ML Model")
        data=pd.read_csv("sourcefile.csv")
        target=st.selectbox("Target",data.columns)
        savepath=st.text_input("Where to save model?")
        model=st.selectbox("Model",['Linear Regression',"Logistic Regression","SVR","SVC","Random Forest Regressor","Random Forest Classifier","Mulltinomial Naive Bayes","Bernoulli Naive Bayes","Gaussian Naive Bayes"])
        skModels=[LinearRegression(),LogisticRegression(),SVR(),SVC(),RandomForestRegressor(),RandomForestClassifier(),MultinomialNB(),BernoulliNB(),GaussianNB()]
        X=data.drop(target,axis=1)
        y=data[target]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
        if model=="Linear Regression":
            index=0
        if model=="Logistic Regression":
            index=1
        if model=="SVR":
            index=2
        if model=="SVC":
            index=3
        if model=="Random Forest Regressor":
            index=4
        if model=="Random Forest Classifier":
            index=5
        if model=="Mulltinomial Naive Bayes":
            index=6
        if model=="Bernoulli Naive Bayes":
            index=7
        if model=="Gaussian Naive Bayes":
            index=8
        m=skModels[index]
        m.fit(X_train,y_train)
        score=m.score(X_test,y_test)
        st.write("Score:"+str(score))
        if st.button("Download Model"):
            joblib.dump(m,savepath)
    else:
        st.error("Source File is not present")
if choice=="Prediction":

    model=st.text_input("Model's path")
    test=st.file_uploader("File to predict")
    if (model is not None) and (test is not None):
        try:
            m=joblib.load(model,"r+")
        except:
            st.error("Write correct path")
        data=pd.read_csv(test)
        try:
             pred=pd.DataFrame(m.predict(data),columns=['Prediction'])
             st.dataframe(pred)
             if st.button("Save Predictions"):
                predpath=st.text_input("Where to save(folder)?")
                if predpath is not None:
                    pred.to_csv(os.path.join(predpath,"prediction.csv"),index=False)
        except:
            st.error("Preprocess your data")
st.sidebar.info("MADE BY AARUSH DIXIT")