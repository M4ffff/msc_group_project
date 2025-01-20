import streamlit as st 


st.markdown("# Page 2 sidebar")
st.sidebar.markdown("# Page 2 sidebar?")

# highlighting:
st.markdown("*Streamlit* is **really** ***cool***.")

# text colours
st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')


# emojis
st.write("Streamlit can do **EMOJIS**: ")
st.markdown("Here's a bouquet &mdash;\
            :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")
st.markdown(":sunglasses:")


# normal writing details
multi = '''If you end a line with two spaces,  
a soft return is used for the next line.

Two (or more) newline characters in a row will result in a hard return.
'''
st.markdown(multi)
