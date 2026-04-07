import streamlit as st 
import scripts.restaurant_helper as restaurant_helper 

st.title("Restaurant Tool")

cuisine = st.sidebar.selectbox("Select a cuisine", ["Italian", "Mexican", "Chinese", "Indian"])

if cuisine: 
    response = restaurant_helper.generate_restaurant_name_and_menu(cuisine=cuisine)
    st.header(f"Restaurant Name: {response['restaurant_name']}")
    st.subheader("Menu:")
    for item in response['menu'].split(", "):
        st.write(f"- {item}")