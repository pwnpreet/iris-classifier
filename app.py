# import basic packages & libraries
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
BASE_DIR= Path(__file__).parent
from streamlit_option_menu import option_menu

#load models
models = {
    "Logistic Regression": joblib.load(BASE_DIR / 'logistic.pkl'),
    "GaussianNB": joblib.load(BASE_DIR / 'gaussian.pkl'),
    "KNN": joblib.load(BASE_DIR / 'knn.pkl'),
    "Decision Tree": joblib.load(BASE_DIR / 'tree.pkl'),
    "SVC": joblib.load(BASE_DIR / 'svc.pkl')
}

corpus= joblib.load( BASE_DIR / 'corpus.pkl')

#load dataset
from sklearn.datasets import load_iris
iris= load_iris()
df= pd.DataFrame(iris.data, columns=iris.feature_names)
x= df
y= iris.target
target_names= iris.target_names

#  to setting page layout, format before loading webpage
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    # layout="wide"
)

# setting a navigation menu 
select = option_menu(
    menu_title=None,
    options=["Home", "Predictions", "Model Visualization", "Chatbot", "Gallery", "About"],
    icons= ["house", "activity", "bar-chart", "robot", "image", "info-circle"],
    orientation="horizontal",
    # add style to navigation menu 
    styles={
        "container": {"padding": "5px", "background-color": "#f8f6fa"},
        "icon": {"color": "#8e44ad", "font-size": "20px"},
        "nav-link": {
            "font-size": "15px",
            "color": "#2c3e50",
            "margin": "0px 10px",
            "text-transform": "capitalize",
        },
        "nav-link-selected": {"background-color": "#8e44ad", "color": "white"},
    },
)

#Home section 
if select == "Home":
    st.title("🪷 Iris Flower Classifier — Know Your Flower 🪷")

    st.image(str(BASE_DIR / "iris.jpg"), use_container_width= True)
    st.markdown("<p class='tagline'> Machine Learning Models: <b>KNN, Logistic Regression, Decision Tree, GaussianNB, SVC</b></p>", unsafe_allow_html=True)
    # adding styling 
    st.markdown(
        """
        <style>
        .tagline {
        text-align: center;
        font-size: 20px;
        color: #555;
        }
        </style>
        """,
        unsafe_allow_html= True
    )
#predictions section 
elif select == "Predictions":
    st.title("🌼 Iris Flower Prediction 🌼")
    st.write("Select Features and Choose a model to predict:")
    # features 
    sepal_len = st.number_input("Enter the sepal length", min_value= 3.0 , max_value= 8.0)
    sepal_wid = st.number_input("Enter the Sepal width", min_value= 2.0, max_value=4.8)
    petal_len = st.number_input("Enter the petal length", min_value= 1.0 , max_value= 7.0)
    petal_wid = st.number_input("Enter the petal width", min_value= 0.1, max_value= 3.0)

    features = [[sepal_len, sepal_wid, petal_len, petal_wid]]
    # choose a model to predict 
    model_name = st.selectbox(
        "Choose a Model",
        ("KNN", "Logistic Regression", "Decision Tree", "GaussianNB", "SVC")
    )

    
    if st.button("🔍 Predict"):
        model = models[model_name]
        prediction = model.predict(features)[0]

        target_names = ['Setosa', 'Versicolor', 'Virginica']
        flower_name = target_names[prediction]

        st.success(f"🌸 The predicted flower is: {flower_name}")

#model visualize section
elif select == "Model Visualization":
    st.title("📊 Model Visualization")
    st.write("Explore how different models perform on Iris dataset.")

    # training the data 
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)
    model_names= st.selectbox("Select model to visualize", list(models.keys()))

    # get model from dictionary
    model= models[model_names]

    y_pred = model.predict(x_test)
    acc= accuracy_score(y_test, y_pred)
    st.metric("Model Accuracy", f"{acc:.2f}") 

    # classification report of model
    st.subheader(f"Classification Report of {model_names} Model")
    st.text(classification_report(y_test, y_pred, target_names=target_names))

    #charts/ plots
    st.subheader(" Visual Insights")

    df_viz= x.copy()
    df_viz['species'] = y
    df_viz['species']=df_viz['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

    st.markdown("**Sepal vs Petal Length** - visualize how species are separated by features")
    fig1= px.scatter(df_viz, x='sepal length (cm)', y= 'petal length (cm)', color= 'species', symbol='species', title= 'Sepal vs Petal Length')
    st.plotly_chart(fig1, use_container_width= True)

    st.markdown("**Petal width distribution** - compare sperad of petal width among species")
    fig2= px.box(df_viz, x='species', y='petal width (cm)', color='species', title='distribution of Petal width')
    st.plotly_chart(fig2, use_container_width= True)

#chatbot section 
elif select == "Chatbot":
    st.title("💬 Iris Chatbot Assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history= []

    user_input= st.text_input("Ask me anything about the Iris dataset or the model used")
    # chatbot response function 
    def chatbot(query):
        query= query.lower()
        for q, a in corpus.items():
            words= q.lower().split()[:3]
            if all(w in query for w in words):
                return a
        return "Sorry, I did not understand."    
        
    if st.button("Ask"):
        response= chatbot(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
        st.success(f"🤖: {response}") 
    #chatbot history
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

 #galery section            
elif select ==  "Gallery":
    st.title("📂 Gallery – Iris Dataset Visualization")
    st.write("Explore the dataset")

    st.subheader("📘 Dataset Preview")
    st.dataframe(df.head()) 

    st.subheader("📐 Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, columns: {df.shape[1]}") 

    st.subheader("📊 Statistics Summary")
    st.write(df.describe())

    st.subheader("🌺 Feature Distribution")
    feature= st.selectbox("select a feature to view distribution", iris.feature_names)
    fig4= px.histogram(df, x= feature, title=f"distribution of {feature}")
    st.plotly_chart(fig4, use_container_width= True)   

#about section
elif select == "About":
    st.title("ℹ️ About This Project")

    st.markdown("""
                🌸 Iris Flower Classification App \n
                This web page is built to explore the **Iris Flower Dataset** using multiple machine learning models. \n
                In this user can: \n
                    🔍 Understand the Dataset \n
                    🐾 Make Predictions \n
                    🤖 Interact with chatbot \n
                    📊 Visualize important features \n
                    🖼️ Explore dataset Gallery
                """)  

    st.subheader("📘 Dataset Information")
    st.markdown("""
                DATASET NAME: Iris Flower Dataset \n
                CREATED BY: Ronald A. Fisher \n
                TOTAL SAMPLES: 150 \n
                FEATURES: 
                  - Sepal Length
                  - Sepal Width
                  - Petal Length
                  - Petal Width \n
                TARGET CLASSES: 
                  - Setosa
                  - Versicolor
                  - Virginica
                """)

    st.subheader("🧠 Model Used in this Project")
    st.markdown("""
                The following machine learning models are used: \n
                K-Nearest Neighbor (KNN) \n
                Logistic Regression \n
                Decision Tree Classifier \n
                Gaussian Naive Bayes \n
                Support Vector Classifier (SVC)
                """)                

    st.subheader("Technologies Used:")
    st.markdown("""
                Python \n
                Streamlit \n
                Scikit-learn \n
                Pandas & Numpy \n
                Plotly & Seaborn \n
                Joblib 
                """)
    
    st.subheader("👩‍💻 Developer")
    st.markdown("""
                DEVELOP BY: Pawanpreet Kaur \n 
                COURSE: Machine Learning & Ai
                """)
    
    st.markdown("---")
    st.markdown("Thanks for exploring this web page! 🤗")


