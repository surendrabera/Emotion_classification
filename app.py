import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import pickle
import spacy

model=pickle.load(open('model', 'rb'))
nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        else:
            filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)
def predict_emotion(text):
    result=model.predict([text])
    return result
def get_probability(text):
    result=model.predict_proba([text])
    return result
def main():
    st.title('Emotion Classification')
    with st.form(key='emotion_cLf_form'):
        text=st.text_area('Type Text Here')
        submit_text=st.form_submit_button(label='submit')
    if submit_text:
        col1,col2=st.columns(2)
        text_t=preprocess(text)
        prediction= predict_emotion(text_t)
        probability=get_probability(text_t)

        with col1:
            st.success('Original Text')
            st.write(text)

            st.success('Prediction')
            st.write(prediction)
            st.write("Confidence: {}".format(np.max(probability)))
        with col2:
            st.success("Probability")
           # st.write(probability)
            proba_df=pd.DataFrame(probability, columns=model.classes_)
            proba_clean=proba_df.T.reset_index()
            proba_clean.columns=['emotions','probability']
            fig=alt.Chart(proba_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ =='__main__':
    main()

