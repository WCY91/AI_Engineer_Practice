import streamlit as st
from ai_agent import get_feedback

def app():
    st.title("Question 5")
    st.write("""
    **How can Tentacle Tents ensure that their decision-making process takes into account the perspectives and preferences of different customer segments, without overemphasizing one particular group?**
    """)

    answer = st.text_area("Your answer:", height=200)

    if st.button("Submit"):
        feedback = get_feedback(answer, 5)
        st.write("**AI Feedback:**")
        st.write(feedback)
