import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
st.set_page_config(page_title = "my dashboard", page_icon = "tada", layout = "wide")

#---Load Assets---


#---Header Section---
with st.container():
    st.subheader("A Customer insight for Ecommerce X :sunglasses:")
    st.title("Customer Information dashboard")
    st.write("This is a University Project developed by IT management Students of Allameh Univ Tehran Iran")
    st.write("[To contact us >](saeid.kazemi7597@gmail.com)")
#---What this page includes---
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("This program includes")
        st.write("##")
        st.write(
            """This dashboard will include some insight and information about the customers of Ecommerce X company icluding:
            - Customer segmentation by the ratio of loyaltiness and stickiness to company
            - geographic distibution of loyal customers
            - distincting the customers by repeated purchases
            - highliting the most desirable goods 
            """
        )
    with right_column:
        chosen = st.radio(
        'Information',
        ("RFM", "Customer loyalty", "customer segmentation",))
    st.write(f"You have {chosen} information!")
    chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"])


st.bar_chart(chart_data)

#---Histogram---
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
hist_data = [x1, x2, x3]
group_labels = ['Group 1', 'Group 2', 'Group 3']
fig = ff.create_distplot(hist_data, group_labels, bin_size = [.1, .25, .5] )
st.plotly_chart(fig, use_container_width= True)

