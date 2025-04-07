import streamlit as st

st.header("Conclusions")

st.markdown("""
## Thank You for Exploring Our ML App!

We're really glad you took the time to interact with this learning tool.  
Hopefully, it's helped you understand some key concepts in **machine learning**, 
from **basic supervised models**, to **neural networks** and **training parameters**.

Whether you're just starting out or already comfortable with these ideas, 
we hope this app made the concepts more accessible and engaging 🚀
""")

# Reflection Slider
st.markdown("### 🌟 Reflect on Your Learning")

st.markdown("**How much do you know about ML *now* after using the app?**")
knowledge_slider_after = st.slider("1 = Beginner to 5 = Expert:", 1, 5, 3)
st.write("Your current knowledge level after completing the app:", knowledge_slider_after)

st.success("👏 Thanks again for exploring machine learning with us!")
st.markdown("*You can always come back to test and build more models!*")

st.link_button("Link to GitHub","https://github.com/M4ffff/msc_group_project")