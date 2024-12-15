import streamlit as st
from ai_agent import get_feedback

def app():
    st.title("Question 1")
    st.write("""
    **How can Tentacle Tents ensure that their evaluation of marketing channels is comprehensive and not solely focused on immediate returns, taking into account long-term benefits and brand image?**
    """)

    answer = st.text_area("Your answer:", height=200)

    if st.button("Submit"):
        feedback = get_feedback(answer, 1)
        st.write("**AI Feedback:**")
        st.write(feedback)
