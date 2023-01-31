from os.path import dirname, join

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from PIL import Image

from ml_app import main

# Setting genérale du Dashbord
st.set_page_config(
    page_title="My Streamlit App",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Mise en place du current path et importation de la base
current_dir = dirname(__file__)
file_path = join(current_dir, "./data/raw/student_data.csv")


options = st.sidebar.radio("Pages", ["Acceuil", "Dashbord "])

if options == "Acceuil":
    st.title("Outils pour détecter les élèves à accompagner")
    st.subheader(
        "Veuillez cliquer sur le bouton pour entrainer le modèle et générer le score"  # noqa
    )
    button1 = st.button("Appuyez ici!")
    if button1:
        ml = main(file_path)

else:
    data = pd.read_pickle("./data/processed/student_data_score.pkl")
    data["StudentID"] = data["StudentID"].apply(str)

    # Définition des fontions pour scatter plot et le box plot
    def build_figure(df: pd.DataFrame, x: str, y: str, color: str) -> go.Figure:  # noqa
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="red",
            color_continuous_scale=px.colors.sequential.Viridis,
            hover_data=["StudentID", "FirstName", "FamilyName"],
        )
        return fig

    def build_box(df: pd.DataFrame, x: str, y: str) -> go.Figure:
        fig = px.box(
            df,
            x=x,
            y=y,
            points="all",
            color_discrete_sequence=["red"],
            hover_data=["StudentID", "FirstName", "FamilyName"],
        )
        return fig

    numeric_vars = data.select_dtypes("int64").columns
    categorial_vars = data.select_dtypes("object").columns

    if len(numeric_vars) < 1:
        st.warning("Aucune variable numérique trouvé")
        st.stop()

    if len(categorial_vars) < 1:
        st.warning("Aucune variable numérique trouvé")
        st.stop()

    image = Image.open("./figures/features_importances.png")
    st.write(
        "Le graphique ci-dessous représente l’importance de chaque variable pour le modèle XGboost:"  # noqa
    )
    st.image(
        image,
        caption="Les 10 top des variables",
        use_column_width="auto",
    )

    col1, col2 = st.columns([2, 0.8])

    with col2:
        st.write("Paramétrage du Scatter plot")
        xvar = st.selectbox("Abscisse Scatterplot", numeric_vars)
        yvar = st.selectbox("Ordonnée Scatterplot", ["score"])
        huevar = st.selectbox(
            "Groupe",
            ["None"]
            + data.drop(["StudentID", "FirstName", "FamilyName"], axis=1)
            .select_dtypes("object")
            .columns.tolist(),
        )
        if huevar == "None":
            huevar = None

    with col1:
        st.subheader("Visualisation des données avec le scatterplot:")
        sc = build_figure(data, x=xvar, y=yvar, color=huevar)
        st.plotly_chart(sc, use_container_width=True, theme="streamlit")

    col3, col4 = st.columns([2, 0.8])
    with col4:
        st.write("Paramétrage du Boxplot")
        xvar = st.selectbox(
            "Abscisse Boxplot",
            data.drop(["StudentID", "FirstName", "FamilyName"], axis=1)
            .select_dtypes("object")
            .columns.tolist(),
        )
        yvar = st.selectbox("Ordonnée Boxplot", ["score"])

    with col3:
        st.subheader("Visualisation des données avec le boxplot:")
        box = build_box(df=data, x=xvar, y=yvar)
        st.plotly_chart(box, use_container_width=True, theme="streamlit")
