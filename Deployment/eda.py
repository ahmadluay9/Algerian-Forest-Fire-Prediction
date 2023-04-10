import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.preprocessing import LabelEncoder

from PIL import Image

st.set_page_config(
    page_title='Predicting Forest Fire in Algeria Using Supervised Learning',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

def run():
    # title
    st.title('Predicting Forest Fire in Algeria Using Supervised Learning')

    # sub header
    st.subheader ('Exploratory Data Analysis of the dataset.')

    # Add Image
    image = Image.open('forest fire.jpg')
    st.image(image,caption = 'Forest fire illustration')

    # Description
    st.write('Forest fires are a serious issue in Algeria, particularly during the summer months when hot, dry weather and strong winds increase the risk of fires spreading rapidly. The country has a significant amount of forested land, particularly in the northern coastal region, which is particularly vulnerable to fires. Predicting forest fires in Algeria is important in order to prevent or mitigate their impact.')
    st.write('# Dataset') 
    st.write('This section explains the process of data loading. Dataset used in this analysis is Algerian Forest Fires Dataset on UCI Machine Learning Repository.')

    # show dataframe
    df = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv')
    st.dataframe(df)
    # add description of Dataset
    st.write('Following are the variables and definitions of each column in the dataset.')
    st.write("`Date` : 	(DD/MM/YYYY) Day, month (june to september), year (2012)")
    st.write("`Temp` : temperature noon (temperature max) in Celsius degrees: 22 to 42")
    st.write("`RH` : 	Relative Humidity in %: 21 to 90")
    st.write("`Ws` : 	Wind speed in km/h: 6 to 29")
    st.write("`Rain` : total day in mm: 0 to 16.8")
    st.write("`FFMC` : Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5")
    st.write("`DMC` : Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9")
    st.write("`DC` : Drought Code (DC) index from the FWI system: 7 to 220.4")
    st.write("`ISI` : Initial Spread Index (ISI) index from the FWI system: 0 to 18.5")
    st.write("`BUI` : Buildup Index (BUI) index from the FWI system: 1.1 to 68")
    st.write("`FWI` : Fire Weather Index (FWI) Index: 0 to 31.1")
    st.write("`Classes` : two classes, namely fire and not fire")

    # Forest Fire

    st.write('# Exploratory Data Analysis ')
    st.write('## Forest Fire')
    st.write('This section describes data exploration to determine the number of forest fires that ocurred in 2012 in Algeria.')
    # number of forest fire
    df_eda = df.copy()
    fire = df_eda.Classes.value_counts().to_frame().reset_index()
    
    # Plot PieChart with Plotly
    fig = px.pie(fire,values='Classes', names='index',color_discrete_sequence=['brown','orange'])
    fig.update_layout(title_text = "Number of Forest Fires")
    st.plotly_chart(fig)
    st.write('From the visualization above, the number of forest fires that occurred is balanced with the number of forest fires that did not occur. After knowing the number of forest fires, further exploration is carried out to find out when forest fires occur most frequently.')

    # The month with the most occurrence of forest fires

    # Number of Forest Fire
    fires = df_eda.groupby(['Classes','month']).aggregate(Number_of_Forest_Fire=('Classes','count'))
    fires = fires.reset_index()

    # plotting bar plot
    fig = px.bar(fires, x="month", y="Number_of_Forest_Fire",color='Classes',color_discrete_sequence=['brown','orange'],
             orientation="v",hover_name="month"        
                
             )
    fig.update_layout(title_text = "Number of forest fires each month")
    st.plotly_chart(fig)
    
    st.write('### The month with the most occurrence of forest fires')
    st.write('This section describes data exploration for finding the `month` in which forest fires occur most frequently.')
    # create bar chart
    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(20, 8))

    sns.countplot(x=df_eda.month, hue=df_eda.Classes,palette='YlOrBr')

    plt.title('Number of forest fires by month')
    plt.xlabel('Month')
    plt.ylabel('Forest Fire Count')
    plt.show()
    st.pyplot(fig)
    st.write('From this dataset it is known that forest fires most often occur in the summer, to be precise in **August**. **August** has the highest average temperature compared to other months, the lowest average relative humidity compared to other months and the lowest average rain intensity compared to other months.')

    
    # Temperature

    st.write('## Temperature')
    st.write('This section describes data exploration to find the `Temperature` when a forest fire occurs.')
    # create bar chart
    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(20, 8))

    sns.countplot(x=df_eda.Temperature, hue=df_eda.Classes,palette='YlOrBr')

    plt.title('Temperature and Forest fires')
    plt.xlabel('Temperature')
    plt.ylabel('Count')
    plt.show()
    st.pyplot(fig)
    st.write('From the visualization above, the average temperature when a forest fire occurs is 33.82 degree Celsius.')
    
    # Relative Humidity

    st.write('## Relative Humidity')
    st.write('This section describes data exploration to find the level of `relative humidity` when a forest fire occurs.')
    # create a new column based on humidity group
    df_eda['RH_bins']=pd.cut(
    x=df_eda['RH'],
    bins=[21,26,31,36,41,46,51,56,61,66,71,76,81,86,91],
    labels=['21-25','26-30','31-35','36-40','41-45','46-50','51-55','56-60','61-65','66-70','71-75','76-80','81-85','86-90'])

    # create bar chart
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 8))

    sns.countplot(x=df_eda.RH_bins, hue=df_eda.Classes,palette='YlOrBr')

    plt.title('Relative Humidity and Forest Fires')
    plt.xlabel('Relative Humidity')
    plt.ylabel('Count')
    plt.show()
    st.pyplot(fig)
    st.write('From the visualization above, forest fires most often occur when the relative humidity is 55%. The more the relative humidity increases, the more likely it is that a forest fire will not occur.')
    
    # Rain

    st.write('## Rain')
    st.write('This section describes data exploration to find the intensity of `rain` when a forest fire occurs.')
    # forest fire when raining
    rain = df_eda[df_eda['Classes']=='fire'].Rain.value_counts().reset_index()
    rain = rain.rename(columns={'index':'rain_intensity'})
    # create bar chart
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 8))

    sns.countplot(x=df_eda[df_eda['Classes']=='fire'].Rain,palette='YlOrBr')

    plt.title('Rain and Forest Fires')
    plt.xlabel('Rain')
    plt.ylabel('Forest Fire Count')
    plt.show()
    st.pyplot(fig)
    st.write('From the visualization above, most forest fire incidents occur when it is not raining. But there are also forest fires that occur when it is raining with low intensity.')

    # Fire Weather Index (FWI) System

    st.write('## Fire Weather Index (FWI) System')
    st.write('This section explains the process of data exploration to find the `FWI` value that can cause forest fire. The Fire Weather Index (FWI) is a numeric rating of fire intensity. It is based on the ISI and the BUI, and __is used as a general index of fire danger throughout the forested areas__.')
    st.write('**Structure of the FWI System**')
    st.write('The diagram below illustrates the components of the FWI System. Calculation of the components is based on consecutive daily observations of `temperature`, `relative humidity`,`wind speed` and `24-hour precipitation`. The six standard components provide numeric ratings of relative potential for wildland fire.')
    # Add Image
    fwi = Image.open('fwi.png')
    st.image(fwi)
    st.write("FWI Range")
    st.write("`0 - 1` : Low ")
    st.write("`2 - 6` : Moderate")
    st.write("`7 - 13` : High")
    st.write("`> 13` : Very High")
    # FWI
    df_eda['FWI_cat']=pd.cut(
    x=df_eda['FWI'],
    bins=[-1,1,6,13,np.inf],
    labels=['Low','Moderate','High','Very High'])

    # create bar chart
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 8))

    sns.countplot(x=df_eda.FWI_cat,hue=df_eda.Classes,palette='YlOrBr')

    plt.title('FWI and Forest Fires')
    plt.xlabel('FWI')
    plt.ylabel('Count')
    plt.show()

    st.pyplot(fig)
    st.write('From the visualization above, the potential for forest fires increases when the FWI value is in the range of 2-6 (moderate).')
    # Number of Forest Fire
    fwi = df_eda.groupby(['FWI_cat','Classes']).aggregate(Number_of_Forest_Fire=('Classes','count'))
    fwi = fwi.reset_index()

    # plotting bar plot
    fig = px.bar(fwi, x="FWI_cat", y="Number_of_Forest_Fire",color='Classes',color_discrete_sequence=['brown','orange'],
             orientation="v",hover_name="FWI_cat"        
                
             )
    fig.update_layout(title_text = "FWI and Forest Fires")
    st.plotly_chart(fig)

    # Correlation Matrix Analysis

    st.write('## Correlation Matrix Analysis')
    st.write('This section explains about correlation matrix analysis to find out the correlation between features and target (`Classes`). The cell below explains the process of performing a correlation matrix analysis to identify the features that are most strongly correlated with the target (`Classes`). To accomplish this, categorical data will be converted into numerical data using the `LabelEncoder` library.')
    df_copy = df.copy()

    # Using LabelEncoder to convert categorical into numerical data
    m_LabelEncoder = LabelEncoder()
    df_copy['Classes']=m_LabelEncoder.fit_transform(df_copy['Classes'])
    df_copy = df_copy.drop(['year'],axis=1)
    # Plotting Correlation Matrix
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(15,15))
    sns.heatmap(df_copy.corr(),annot=True,cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)
    st.write('From the visualization above, `month` and `Ws` has a low correlation to the target (`Classes`).')

if __name__ == '__main__':
    run()