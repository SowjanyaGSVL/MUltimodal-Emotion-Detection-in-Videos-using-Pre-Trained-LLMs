import streamlit as st
from introduction import display_introduction # Import the function

# Define the options for the tabs
app_options = {
    "Introduction": display_introduction,  # Link to the function
    "Multimodal": r"E:\final notebook\Notebooks\app2.py",
    "Text model": r"E:\final notebook\Notebooks\Text\app.py",
    "Audio Model": r"E:\final notebook\Notebooks\Audio\app.py"
}

# Apply custom CSS to style the tab labels
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 28px;  /* Adjust the font size as needed */
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create tabs for each app 
tabs = st.tabs(list(app_options.keys()))

# Iterate through tabs and display the content
for tab_name, tab in zip(app_options.keys(), tabs):
    with tab:
        if tab_name == "Introduction":
            app_options[tab_name]()  # Call the function 

        else:
            st.write(f"Loading {tab_name}...")
            app_file = app_options[tab_name]

            try:
                exec(open(app_file).read())
            except Exception as e:
                st.error(f"An error occurred while loading {tab_name}: {e}")