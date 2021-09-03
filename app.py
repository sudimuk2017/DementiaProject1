import streamlit as st
import tensorflow as tf
import numpy as np


#loading the trained model
model = tf.keras.models.load_model('model.h5')

@st.cache(allow_output_mutation=True)

def prediction(Age,Gender,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF):
    
    # preprocess user input
    if Gender == 'Male':
        Gender = 0
    else:
        Gender = 1
        
    # making predictions
    predictions = model.predict(
        [[Age,Gender,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF]])
        
    score = tf.nn.softmax(predictions[0])
    class_names = ['Converted','Demented','Nondemented']
    final_pred = class_names[np.argmax(predictions)]
        
    return final_pred
        
# main function defines our webpage

def main():
    
     # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">ALzeimers Hub</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    # allow user input 
    Age  = st.number_input('Age',min_value=1, max_value=100, value=10, step=1)
    Gender = st.selectbox('Gender',('Male','Female'))
    EDUC  = st.number_input('Years of Education: EDUC',min_value=1, max_value=100, value=1, step=1)
    SES  = st.number_input('Socialeconomic Status (SES): 1-5',min_value=1, max_value=50, value=1, step=1)
    MMSE = st.number_input('Mini Mental State Examination (MMSE)',min_value=1, max_value=40, value=1, step=1)
    CDR = st.number_input('Clinical Dimentia Rating (CDR): 0-3',min_value=0.0, max_value=3.0, value=0.0, step=0.5)
    eTIV = st.number_input(' Estimated total intracranial volume:eTIV ',min_value=1000, max_value=2000, value=1000, step=1)
    nWBV  = st.number_input('Normalized whole Brain Volume: nWBV',min_value=0.0, max_value=1.0, value=0.0,step=0.01)
    ASF  = st.number_input('Atlas Scaling Factor: ASF',min_value=0.7, max_value=1.6, value=0.7, step=0.01)
    
    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(Age,Gender,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF)
        
        st.success('Health Status is {}'.format(result))
        
if __name__=='__main__': 
    main()
