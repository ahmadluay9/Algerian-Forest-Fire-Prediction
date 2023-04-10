import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
# Load All Files

with open('prepmod_dt.pkl', 'rb') as file_1:
  prepmod_dt = pickle.load(file_1)

with open('Drop_Columns.txt', 'r') as file_2:
  Drop_Columns = json.load(file_2)

def run():
  with st.form(key='form_forest_fire'):
      day = st.slider('Enter Date',min_value=1,max_value=31,value=26)
      month = st.slider('Enter Month',min_value=1, max_value=12,value=7)
      year = st.number_input('Enter Year',min_value=2012,max_value=2012,value=2012)
      st.markdown('---')
      Temperature = st.number_input('Enter Temperature in degree Celsius',min_value=22,max_value=42,value=36)
      RH = st.number_input('Enter RH (Relative Humidity) in %',min_value=21,max_value=90,value=53)
      Ws = st.number_input('Enter Wind speed in km/h',min_value=6,max_value=29,value=19)
      Rain = st.number_input('Enter Rainfall in mm',step=0.01,format="%.2f",min_value=0.00,max_value=16.80,value=0.00)
      st.markdown('---')
      FFMC = st.number_input('Fine Fuel Moisture Code (FFMC)',step=0.1,format="%.2f",min_value=28.60,max_value=92.50,value=89.20)
      DMC = st.number_input('Duff Moisture Code (DMC)',step=0.1,format="%.2f",min_value=1.10,max_value=65.90,value=17.10)
      DC = st.number_input('Drought Code (DC)',step=0.1,format="%.2f",min_value=7.00,max_value=220.40,value=98.60)
      ISI = st.number_input('Initial Spread Index (ISI)',step=0.1,format="%.2f",min_value=0.00,max_value=18.50,value=10.00)
      BUI = st.number_input('Buildup Index (BUI)',step=0.1,format="%.2f",min_value=1.10,max_value=68.00,value=23.90)
      FWI = st.number_input('Fire Weather Index (FWI)',step=0.1,format="%.2f",min_value=0.00,max_value=31.10,value=15.30)
       
      submitted = st.form_submit_button('Is there a forest fire?')

  df_inf = {
      'day': day,
      'month': month,
      'year': year,
      'Temperature': Temperature,
      'RH': RH,
      'Ws': Ws,
      'Rain': Rain,
      'FFMC': FFMC,
      'DMC': DMC,
      'DC': DC,
      'ISI': ISI,
      'BUI':BUI,
      'FWI':FWI
  }
  df_inf = pd.DataFrame([df_inf])
  # Data Inference
  df_inf_copy = df_inf.copy()
  

  # Removing unnecessary features
  df_inf_final = df_inf_copy.drop(Drop_Columns,axis=1).sort_index()
  
  st.dataframe(df_inf_final)

  if submitted:
      # Predict using DecisionTree
      y_pred_inf = prepmod_dt.predict(df_inf_final)
      st.write('# Is there a forest fire?')
      if y_pred_inf == 0:
         st.subheader('There is No Forest Fire')
      else:
         st.subheader('There is a Forest Fire')

if __name__ == '__main__':
    run()