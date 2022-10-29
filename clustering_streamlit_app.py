## pipreqs (to create requirement file needed for streamshare)
## cd to folder
## streamlit run clustering_streamlit_app.py
########################## Initialization #####################

import streamlit as st

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
import pickle

import warnings
import os
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('max_colwidth', 150)
st.set_page_config(layout="wide")


@st.cache
def load_data():
	
	df1 = pd.read_csv(r'Mall_Customers.csv')
	Data = pd.read_csv ("labelledData.csv")  
	# Dbscan = pickle.load(open("dbscan.pickle", 'rb'))
	# kmeans = pickle.load(open("kmeans.pickle", 'rb'))
	Dbscan = pickle.load(open('db.pkl', 'rb'))
	kmeans = pickle.load(open('kmeans.pkl', 'rb'))
	return df1,Data

df1,Data= load_data()
df1 = df1.drop('CustomerID',axis=1)



"Hi"
# ###################### Data Calculations #####################
fig_num = 1
df = Data



@st.cache(allow_output_mutation=True) # hash_funcs because dict can't be hashed
def load_plotly_fig():

	data_kmeans = [1]
	data_db = [1]
	data_layout = [1]
# 	df = Data
	
# 	st.write("hello")
# 	dbtrace = go.Scatter3d(
# 	    x= df['Age'],
# 	    y= df['Spending Score (1-100)'],
# 	    z= df['Annual Income (k$)'],
# 	    mode='markers',
# 	    name= "DBscan - original data",
# 	     marker=dict(
# 	        color = -df['label4'], 
# 	        size= 10,
# 	        line=dict(
# 	            color= -df['label4'],
# 	            width= 12
# 	        ),
# 	        opacity=0.8
# 	     )
# 	)

# 	kmeanstrace = go.Scatter3d(
# 	    x= df['Age'],
# 	    y= df['Spending Score (1-100)'],
# 	    z= df['Annual Income (k$)'],
# 	    mode='markers',
# 	    name= "Kmeans - original data",
# 	     marker=dict(
# 	        color = df['label3'], 
# 	        size= 10,
# 	        line=dict(
# 	            color= df['label3'],
# 	            width= 12
# 	        ),
# 	        opacity=0.8
# 	     )
# 	)

# 	layout = go.Layout(

# 	#     )
# 		height=600,
# 		template="plotly_dark",
# 	    #title= 'DBScan Clusters',
# 	    scene = dict(
# 	            xaxis = dict(title  = 'Age'),
# 	            yaxis = dict(title  = 'Spending Score'),
# 	            zaxis = dict(title  = 'Annual Income')
# 	        )
# 	)

# 	data_kmeans = [kmeanstrace]
# 	data_db = [dbtrace]
# 	data_layout = [layout]


	
	return data_kmeans,data_db,data_layout

data_kmeans,data_db,data_layout = load_plotly_fig()

# fig_kmeans = go.Figure(data=data_kmeans, layout=data_layout[0])
# fig_dbscan = go.Figure(data=data_db, layout=data_layout[0])
# ### EDA Figures


# @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
# def mk_figure():
# 	fig_num = 1
# 	parameter_distribution = plt.figure(fig_num , figsize = (15 , 6))
# 	n = 0 
# 	for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
# 	    n += 1
# 	    plt.subplot(1 , 3 , n)
# 	    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
# 	    sns.distplot(df[x] , bins = 20)
# 	    plt.title('Distplot of {}'.format(x))

# 	fig_num +=1


	
# 	gender_plot = plt.figure(fig_num , figsize = (15 , 5))
# 	sns.countplot(y = 'Gender' , data = df)	
# 	fig_num +=1


# 	pairplot = sns.pairplot(df1)



# 	return parameter_distribution,gender_plot,pairplot

# parameter_distribution,gender_plot,pairplot = mk_figure()

# ########################## Page UI #####################

# ### Overview of Project
# tab1, tab2, tab3 = st.tabs([ "WebApp","Project Overview", "Methodology"])

# with tab1:
# 	" # Web App "
# 	" This web app shows you clustering done by two algorithms for customer segmentation of a mall"
# 	" Even though this is not an implementation of classification algorithm, you can play with the input parameters to see where does the customer stands in comaparison with other customers"
# 	"---"
# 	col_1, col_2, col_3 = st.columns(3)

# 	with col_1:
# 		#name = st.text_input("What is your name ?")
# 		Spending_Score = st.slider('Customer Spending Score', 0, 100, 35)
# 	with col_2:
# 		age = st.slider('Age of the Customer? ', 15, 70, 35)		
# 	with col_3:
# 		income = st.slider('Annual income of the customer in k$?', 15, 135, 60)	

# 	col_10, col_20 = st.columns(2)
# 	current_point = go.Scatter3d(
# 	    x= [age],
# 	    y= [Spending_Score],
# 	    z= [income],
# 	    mode='markers',
# 	    name= "Your Selection",
# 	     marker=dict(
# 	        color = "orange", 
# 	        size= 20,
# 	        opacity=1
# 	     )
# 	)


# 	fig_dbscan.add_trace(current_point)
# 	fig_kmeans.add_trace(current_point)

# 	with col_10:
# 		'### Kmeans'
# 		" Yellow markers are highly valued customers - young, high earning with high spending score"
# 		st.plotly_chart(fig_kmeans, use_container_width=True)
# 	with col_20:
# 		'### DBscan'
# 		" Purple markers are highly valued customers - young, high earning with high spending score"
# 		st.plotly_chart(fig_dbscan, use_container_width=True)





# with tab2:
# 	"## Project Overview"
# 	"This is the first project in the ML for Data science series. The aim of this project is to"
# 	"- Explore basic python and ML libraries - to perfrom Data manipulation, visualization, Exploratory Data Analysis and model training/building"
# 	"- Explore basic clustering algo like kmeans and DBscan"
# 	"- Explore how to build streamlit app and deploy trained models on heroku server."

# 	st.info(' Main Aim is to understand how the data science pipeline works and get used to basic tools, and not to build accurate model')

# 	"---"
# 	'## Sources'
# 	"[Kaggle DataSet](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)"
# 	"[Github](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)"

# 	# "---"
# 	# "- Checkout the app in WebApp tab"
# 	# "- Checkout the step by step analysis in Methodology tab"
# 	"---"

# 	" ## Other Projects on this series"
# 	" 2. [An Internal Link](/guides/content/editing-an-existing-page)"
# 	" 3. [An Internal Link](/guides/content/editing-an-existing-page)"
# 	" 4. [An Internal Link](/guides/content/editing-an-existing-page)"

# with tab3:

# 	"# Problem Statement"



# 	"We want to segment highspending value customers for the mall. As data is just 3D and highly intuitive, it is common sense that we need to target younger people with high salary and high spending score."
# 	'Building ML model for this data and problem doesnt make sense- but the aim is not to build accurate model. Aim is to understand how the data science pipeline works and get used to basic tools. Lets see if our model can cluster these highly imp customers.'
# 	'The main aim of the projects is to get understanding of all the basic libraries required to perform EDA and Feature Engineering, ML model building/Training, developing and deploying web app. As with the successful clustering,plotting,developing streamlit app and deploying it on the server has acheived this purpose'

# 	"---"
# 	"#### DataSet"
# 	st.table(df1.head())
# 	st.write("Shape of the data" , df1.shape)

# 	'### Exploratory Data Analysis'
# 	col_10, col_20 = st.columns([3, 1])
# 	with col_10:
# 		st.pyplot(parameter_distribution,use_container_width=True)

# 	with col_20:
# 		" Distribution plot of all three numeric parameters signify that the data is normally distributed"

# 	col_10, col_20 = st.columns([3, 1])
# 	with col_10:
# 		st.pyplot(gender_plot)

# 	with col_20:
# 		" We understand that the numbber of Female are more than that of male but still the data is fairly balanced"		
	

# 	col_10, col_20 = st.columns([3, 1])
# 	with col_10:
# 		st.pyplot(pairplot)
# 	with col_20:
# 		" Correlation plot does not reveal any correlation between any features"




# 
