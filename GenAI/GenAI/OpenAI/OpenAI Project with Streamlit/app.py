import streamlit as st

st.set_page_config(page_title="Tentacle Tents Decision-Making Case Study", layout="wide")

# Dictionary mapping page names to their respective modules
PAGES = {
    "Question 1": "page1",
    "Question 2": "page2",
    "Question 3": "page3",
    "Question 4": "page4",
    "Question 5": "page5",
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page_module = PAGES[selection]

# Dynamically import the selected page module
page = __import__(page_module)
page.app()
