# Core Pkgs
import streamlit as st

# EDA Pkgs
import numpy as np
import pandas as pd

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():

    #df = load_data()
    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    if data is not None:
        df = pd.read_csv(data)
        page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Plots', 'Prediction'])

        if page == 'Homepage':
            st.title('Wine Alcohol Class Prediction')
            st.text('Select a page in the sidebar')
            st.dataframe(df)
        elif page == 'Exploration':
            st.title('Explore the Wine Data-set')
            if st.checkbox("Show Shape"):
                st.dataframe(df.shape)

            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.dataframe(all_columns)

            if st.checkbox('Show column descriptions'):
                st.dataframe(df.describe())

            if st.checkbox("Show Selected Columns"):
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Show Value Counts"):
                st.dataframe(df.iloc[:,-1].value_counts())
                        
            st.markdown('### Analysing column relations')
            st.text('Correlations:')
            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(df.corr(), annot=True, ax=ax)
            st.pyplot()
            st.text('Effect of the different classes')
            sns.pairplot(df, vars=['magnesium', 'flavanoids', 'nonflavanoid_phenols', 'proline'], hue='alcohol')
            st.pyplot()

        elif page == 'Plots':
            st.subheader("Data Visualization")
            st.title('Plots')
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
                st.pyplot()
            # Customizable Plot
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
            selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

            if st.button("Generate Plot"):
                st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
                # Plot By Streamlit
                if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)
                elif type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)
                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)
                # Custom Plot
                elif type_of_plot:
                    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                    st.write(cust_plot)
                    st.pyplot()

        else:
            st.title('Modelling')
            model, accuracy = train_model(df)
            st.write('Accuracy: ' + str(accuracy))
            st.markdown('### Make prediction')
            st.dataframe(df)
            row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)
            st.markdown('#### Predicted')
            st.text(model.predict(df.drop(['alcohol'], axis=1).loc[row_number].values.reshape(1, -1))[0])


@st.cache(allow_output_mutation=True)
def train_model(df):
    X = np.array(df.drop(['alcohol'], axis=1))
    y= np.array(df['alcohol'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, model.score(X_test, y_test)

#@st.cache
#def load_data():
 #   return pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols','flavanoids', 'nonflavanoid_phenols' ,'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline'], delimiter=",", index_col=False)


if __name__ == '__main__':
    main()