import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import BernoulliNB,GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pycaret.classification import setup as setup_clf, compare_models as compare_clf,pull as pull_clf,save_model as save_clf,load_model as load_clf,predict_model as predict_clf
from pycaret.regression import setup as setup_reg, compare_models as compare_reg,pull as pull_reg,save_model as save_reg,load_model as load_reg,predict_model as predict_reg
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
        le_col=st.selectbox("Column you want to encode?",data.columns)
        if st.button("Label Encoder"):
            le=LabelEncoder()
            data[le_col]=le.fit_transform(data[le_col])
            st.dataframe(le.fit_transform(data[le_col]))
            data.to_csv("sourcefile.csv")
    else:
        st.error("Firstly upload your file")
if choice=="ML Model":
    if os.path.exists("sourcefile.csv"):
        st.header("Select your target and ML Model")
        data=pd.read_csv("sourcefile.csv")
        target=st.selectbox("Target",data.columns)
        clf_or_reg=st.selectbox("Classification or Regression",['Classification',"Regression"])
        savepath=st.text_input("Where to save model?")
        if st.button("Build"):
            best_model=None
            if clf_or_reg=="Classification":
                exp_clf=setup_clf(data,target=target)
                st.header("Setup")
                exp_clf=pull_clf()
                st.dataframe(exp_clf)
                best_model=compare_clf()
                model_df=pull_clf()
                st.header("Models(Ranking Based)")
                st.dataframe(model_df)
            elif clf_or_reg=="Regression":
                exp_reg=setup_reg(data,target=target)
                st.header("Setup")
                exp_reg=pull_reg()
                st.dataframe(exp_reg)
                best_model=compare_reg()
                model_df=pull_reg()
                st.header("Models(Ranking Based)")
                st.dataframe(model_df)
            if st.button("Download"):
                if clf_or_reg=="Classification":
                    save_clf(best_model,savepath)
                elif clf_or_reg=="Regression":
                    save_reg(best_model,savepath)
    else:
        st.error("Source File is not present")
if choice=="Prediction":

    model_path=st.text_input("Model's path")
    test=st.file_uploader("File to predict")
    if (model_path is not None) and (test is not None):
        try:
            model=None
            clf_or_reg=st.selectbox("Classification or Regression",['Classification',"Regression"])
            if clf_or_reg=="Classification":
                model=load_clf(model_path)
            elif clf_or_reg=="Regression":
                model=load_reg(model_path)
        except Exception as e:
            print(e)
            st.error("Write correct path")
        data=pd.read_csv(test)
        try:
            if clf_or_reg=="Classification":
                pred=pd.DataFrame(predict_clf(model,data),columns=['Prediction'])
            elif clf_or_reg=="Regression":
                pred=pd.DataFrame(predict_reg(model,data),columns=['Prediction'])
            st.dataframe(pred)
            if st.button("Save Predictions"):
                predpath=st.text_input("Where to save(?")
                if predpath is not None:
                    pred.to_csv(predpath,index=False)
        except Exception as e:
            print(e)
            st.error("Preprocess your data")
st.sidebar.info("MADE BY AARUSH DIXIT")
