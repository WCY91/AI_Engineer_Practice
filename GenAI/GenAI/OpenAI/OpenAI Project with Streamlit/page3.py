import streamlit as st
from ai_agent import get_feedback

def app():
    st.title("Question 3")
    st.write("""
    **What strategies could Tentacle Tents employ to avoid becoming overly committed to past decisions or investments that may not be yielding the desired results?**
    """)

    answer = st.text_area("Your answer:", height=200)

    if st.button("Submit"):
        feedback = get_feedback(answer, 3)
        st.write("**AI Feedback:**")
        st.write(feedback)
