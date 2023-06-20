import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt 



# Load the LSTM model
pickle_in = open("lstm model.pkl","rb")
model=pickle.load(pickle_in)



# Function to make predictions
def make_predictions(input_data):
    x_input = np.array([187, 196, 210])
    temp_input=list(x_input)
    output=[]
    i=0

    while(i<input_data):
    
        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            #print(x_input)
            x_input = x_input.reshape((1, 3, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            #print(temp_input)
            output.append(yhat[0][0])
            i=i+1
        else:
            x_input = x_input.reshape((1, 3, 1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.append(yhat[0][0])
            output.append(yhat[0][0])
            i=i+1
        
    # data visualization
    day_new=np.arange(1,10)
    day_pred=np.arange(10,input_data+10)

    # plotting 
    timeseries_data = [110, 143 , 158, 17125, 133, 1462, 187, 196, 210]
    plt.plot(day_new,timeseries_data)
    #plt.plot(day_pred,output)
    plt.show()


    print(output)
    return output

# Streamlit app
def main():
    # Set the title and layout of the app
    st.title('LSTM Model Deployment ')
    st.write('Enter your input data:')

    # Get user input
    
    data = st.number_input('Enter the number of months to forecast', min_value=1, value=3)

    if st.button('Predict'):
        # Make predictions using the LSTM model
        result = make_predictions(data)
        st.dataframe(result)  # diplay prediction in table

    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        st.text("Built by satwik")
    st.markdown('---')
    st.markdown('Developed by Satwik')

if __name__ == '__main__':
    main()
