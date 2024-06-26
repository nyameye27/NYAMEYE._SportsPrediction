import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="RuvaS20/Training-Model", filename="xgb_model.joblib")
model = joblib.load(model_path)

# Main program for Streamlit to use
def main():
    st.title("FIFA Player Rating Predictor")
    html_temp = """
    <div style="background:#ADD8E6; padding:10px">
    <h2 style="color:black; text-align:center;">Sports Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    wage_eur = st.number_input('Player Wage in Euros')
    value_eur = st.number_input('Player Value in Euros')
    age=st.number_input('Age in Years')
    potential = st.number_input('Player Potential out of 100', 1, 100, 1)
    defending = st.number_input('Defending out of 100',1, 100, 1)
    defending_standing_tackle=st.number_input('Defending standing tackle out of 100',1,100,1)
    mentality_interceptions=st.number_input('Mentatility of Player out of 100', 1, 100,)
    defending_sliding_tackle=st.number_input('Defending sliding tackle out of 100',1,100,1) 
    movement_reactions=st.number_input('Movement reactions out of 100',1,100,1)
    attacking_finishing=st.number_input('Attacking finish out of 100',1,100,1)

   
    
    

    if st.button('Predict'):
        data = {
            'wage_eur': [wage_eur],
            'value_eur': [value_eur],
            'age':[age],
            'potential': [potential],
            'defending':[defending],
            'defending_standing_tackle':[defending_standing_tackle], 
            'mentality_interceptions':[mentality_interceptions],
            'defending_sliding_tackle':[defending_sliding_tackle], 
            'movement_reactions':[movement_reactions], 
            'attacking_finishing':[attacking_finishing]
        }

        # Making into a DataFrame
        df = pd.DataFrame(data)

        # Ensure the DataFrame has the same columns as the model expects
        expected_features = [
            'wage_eur', 'value_eur','age', 'potential','defending','defending_standing_tackle', 'mentality_interceptions','defending_sliding_tackle', 'movement_reactions', 
            'attacking_finishing'
        ]
        
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # or some default value

        df = df[expected_features]  # Reorder columns to match model's expectation

        prediction = model.predict(df)
        st.write("The predicted overall for your player is ", prediction[0])

if __name__ == '__main__':
    main()
