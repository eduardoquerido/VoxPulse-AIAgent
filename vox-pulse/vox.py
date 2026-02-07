import sys
import os
from dotenv import load_dotenv

load_dotenv()

try:
    import sqlite3

    import pysqlite3

    if sqlite3.sqlite_version_info < (3, 35, 0):
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        print("SQLite3 patched successfully with pysqlite3-binary")
except ImportError:
    print("DEBUG: pysqlite3 not found in sys.path. Still using system sqlite3.")

import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from voxpulse import run_analysis

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

LANG_MAP = {
    "Português-BR": {
        "sidebar_settings": "Configurações",
        "lang_label": "Idioma:",
        "main_input_label": "Nome do Político Principal:",
        "comp_input_label": "Adicionar Candidatos para Comparação:",
        "btn_analyze": "Analisar Pulso Digital",
        "exec_summary": "📋 Resumo Executivo",
        "graph_section": "📊 Comparação de Métricas",
        "radar_title": "Perfil Multidimensional",
        "metrics": {
            "sentiment_score": "Percepção Pública",
            "economic_trust": "Confiança Econômica",
            "digital_presence": "Presença Digital",
            "social_approval": "Aprovação Social",
            "public_security": "Segurança Pública",
        },
    },
    "English": {
        "sidebar_settings": "Settings",
        "lang_label": "Language:",
        "main_input_label": "Main Politician Name:",
        "comp_input_label": "Add Candidates for Comparison:",
        "btn_analyze": "Analyze Digital Pulse",
        "exec_summary": "📋 Executive Summary",
        "graph_section": "📊 Metric Comparison",
        "radar_title": "Multidimensional Profile",
        "metrics": {
            "sentiment_score": "Public Perception",
            "economic_trust": "Economic Trust",
            "digital_presence": "Digital Presence",
            "social_approval": "Social Approval",
            "public_security": "Public Security",
        },
    },
}

st.set_page_config(page_title="VoxPulse-AI", page_icon="🗳️", layout="wide")

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("Language / Idioma", options=["Português-BR", "English"])
    t = LANG_MAP[language]

    st.divider()
    st.subheader(t["comp_input_label"])
    # Let users choose or type new candidates
    candidates = st.multiselect(
        "Candidates:",
        options=[
            "Ciro Gomes",
            "Eduardo Leite",
            "Flávio Bolsonaro",
            "Luís Inácio 'Lula' da Silva",
            "Michele Bolsonaro",
            "Romeu Zema",
            "Renan Santos",
            "Ronaldo Caiado",
            "Ratinho Júnior",
            "Tarcísio de Freitas",
        ],
        default=[],
    )
    st.info("The main politician will automatically be included in the graphs.")

st.title("🗳️ VoxPulse-AI")
st.markdown(f"**Date:** {datetime.date.today()}")

politician = st.text_input(
    t["main_input_label"], placeholder="e.g., President of Brazil"
)


# Analysis Block
# 1 hour cache for Free APIs
@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_analysis(politician_name, candidates_list, lang):
    all_names_str = ", ".join([politician_name] + candidates_list)
    results = run_analysis(politician_name, all_names_str, lang)

    graph_data = None
    try:
        if results.json_dict:
            graph_data = results.json_dict
        else:
            for task_output in reversed(results.tasks_output):
                if task_output.json_dict:
                    graph_data = task_output.json_dict
                    break
    except Exception as e:
        print(f"Error while extraxting JSON {e}")
    
    try:
        report_text = (
            results.tasks_output[1].raw
            if len(results.tasks_output) > 1
            else results.raw
        )
    except:
        report_text = results.raw

    return {
        "raw_report": report_text,
        "graph_json": graph_data,
        "main_politician": politician_name,
    }


if st.button(t["btn_analyze"]):
    if not politician:
        st.warning("Please enter a name of a politician.")
    else:
        with st.status(f"Analyzing {politician}...", expanded=True) as status:
            try:

                cached_data = get_cached_analysis(politician, candidates, language)

                st.session_state.analysis_data = cached_data

                status.update(
                    label=(
                        "Analysis Complete (from cache)!"
                        if st.session_state.analysis_data
                        else "Analysis Complete!"
                    ),
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                st.error(f"Error: {e}")

## Results Block
if st.session_state.analysis_data:
    data = st.session_state.analysis_data

    st.divider()

    col_text, col_graphs = st.columns([1.2, 1])

    with col_text:
        st.subheader(f"{t['exec_summary']} - {data['main_politician']}")
        st.markdown(data["raw_report"])

        if st.button(
            "Limpar Resultados" if language == "Portuguese-BR" else "Clear Results"
        ):
            st.session_state.analysis_data = None
            st.rerun()

    with col_graphs:
        st.subheader(t["graph_section"])

        if data.get("graph_json") is None:
            st.error("The model couldn't return a valid JSON format")

            if st.checkbox("See JSON response"):
                st.write(data.get("raw_report"))
        
        
        elif  "results" in data["graph_json"]:
            df = pd.DataFrame(data["graph_json"]["results"])

            ################ Bar Graph #####################
            fig_bar = px.bar(
                df,
                x="name",
                y="sentiment_score",
                color="name",
                title=t["metrics"]["sentiment_score"],
                labels={
                    "sentiment_score": t["metrics"]["sentiment_score"],
                    "name": "Politician",
                },
                template="plotly_dark",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()

            ################# Radar Graph ###################

            df_radar = df.rename(columns=t["metrics"])
            metrics_cols = list(t["metrics"].values())

            sentiment_label = t["metrics"]["sentiment_score"]
            if sentiment_label in metrics_cols:
                metrics_cols.remove(sentiment_label)

            df_melted = df_radar.melt(
                id_vars=["name"],
                value_vars=metrics_cols,
                var_name="Metric",
                value_name="Value",
            )

            fig_radar = px.line_polar(
                df_melted,
                r="Value",
                theta="Metric",
                color="name",
                line_close=True,
                range_r=[0, 100],
                title=t["radar_title"],
                template="plotly_dark",
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Structured data for charts was not generated in this analysis. The textual report is available on the side.")
            if st.checkbox("See technical errors"):
                st.write(data.get("graph_json"))

st.divider()
st.caption("VoxPulse-AI POC - Developed for the 2026 Election Portfolio.")
