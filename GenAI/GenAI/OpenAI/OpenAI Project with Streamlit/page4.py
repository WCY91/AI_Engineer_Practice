import streamlit as st
from ai_agent import get_feedback

def app():
    st.title("Question 4")
    st.write("""
    **What factors should Tentacle Tents consider in their evaluation of what would be effective for their own brand, and how can they maintain an independent perspective?**
    """)

    answer = st.text_area("Your answer:", height=200)

    if st.button("Submit"):
        feedback = get_feedback(answer, 4)
        st.write("**AI Feedback:**")
        st.write(feedback)
