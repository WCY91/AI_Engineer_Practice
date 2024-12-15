import streamlit as st
from ai_agent import get_feedback

def app():
    st.title("Question 2")
    st.write("""
    **How might Monika Prpic's intuition affect her decision-making process in determining the most effective marketing channels? What methods can be used to validate or complement intuitive decision-making?**
    """)

    answer = st.text_area("Your answer:", height=200)

    if st.button("Submit"):
        feedback = get_feedback(answer, 2)
        st.write("**AI Feedback:**")
        st.write(feedback)
