"""
=============================================================================
Phase 4: Interactive Plotly Dash Dashboard - Mumbai REIT Valuation Engine
=============================================================================
Mimics a Power BI-style dashboard with:
  1. Mumbai Ward Map (Scatter Mapbox)
  2. SHAP Feature Importance Bar Chart (per cluster)
  3. Yield vs Price Scatter Plot
  4. Cluster Distribution & KPIs
=============================================================================
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── Ward coordinates (approximate centroids for Mumbai) ───────────────────────
WARD_COORDS = {
    "South Mumbai (A Ward)": (18.9322, 72.8347),
    "Bandra (H-West Ward)": (19.0596, 72.8295),
    "Andheri (K-West Ward)": (19.1197, 72.8464),
    "Powai (S Ward)": (19.1176, 72.9060),
    "Thane (Beyond BMC)": (19.2183, 72.9781),
    "Worli (G-South Ward)": (19.0100, 72.8172),
    "Borivali (R-North Ward)": (19.2307, 72.8567),
    "Navi Mumbai": (19.0330, 73.0297),
}

CLUSTER_COLORS = {"Value": "#22c55e", "Premium": "#3b82f6", "Luxury": "#f59e0b"}


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "mumbai_realestate_clustered.csv"))
    # Add lat/lng for map
    df["Latitude"] = df["Ward"].map(lambda w: WARD_COORDS.get(w, (19.08, 72.88))[0])
    df["Longitude"] = df["Ward"].map(lambda w: WARD_COORDS.get(w, (19.08, 72.88))[1])
    # Jitter coordinates for visual spread
    df["Latitude"] += np.random.normal(0, 0.008, len(df))
    df["Longitude"] += np.random.normal(0, 0.008, len(df))
    return df


def load_shap_importance():
    path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


def load_metrics():
    path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"metrics": {"silhouette_score": "N/A", "davies_bouldin_index": "N/A"}}


def create_app():
    df = load_data()
    shap_df = load_shap_importance()
    metrics = load_metrics()

    app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
               meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
    app.title = "Mumbai REIT Valuation Engine"

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    def kpi_card(title, value, color="#3b82f6"):
        return dbc.Card(dbc.CardBody([
            html.H6(title, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(value, style={"color": color, "fontWeight": "bold"}),
        ]), className="shadow-sm", style={"backgroundColor": "#1e293b", "border": "none"})

    total_props = len(df)
    avg_price = f"₹{df['Sale_Price_INR'].mean()/1e7:.1f} Cr"
    avg_yield = f"{df['Investment_Yield_Pct'].mean():.2f}%"
    sil_score = metrics.get("metrics", {}).get("silhouette_score", "N/A")
    db_index = metrics.get("metrics", {}).get("davies_bouldin_index", "N/A")

    # ── Layout ────────────────────────────────────────────────────────────────
    app.layout = dbc.Container([
        # Header
        dbc.Row(dbc.Col(html.Div([
            html.H2("🏙️ Mumbai REIT Valuation Engine", className="mb-0",
                     style={"color": "#f8fafc", "fontWeight": "700"}),
            html.P("Data Warehouse & Mining Dashboard | K-Means Clustering + SHAP XAI",
                   style={"color": "#94a3b8", "fontSize": "0.9rem"}),
        ]), width=12), className="mb-3 mt-3"),

        # KPIs
        dbc.Row([
            dbc.Col(kpi_card("Total Properties", f"{total_props:,}"), md=2),
            dbc.Col(kpi_card("Avg Sale Price", avg_price, "#22c55e"), md=2),
            dbc.Col(kpi_card("Avg Yield", avg_yield, "#f59e0b"), md=2),
            dbc.Col(kpi_card("Silhouette Score", str(sil_score), "#8b5cf6"), md=2),
            dbc.Col(kpi_card("Davies-Bouldin", str(db_index), "#ef4444"), md=2),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Select Cluster", className="text-muted mb-1", style={"fontSize": "0.8rem"}),
                dcc.Dropdown(id="cluster-filter",
                    options=[{"label": "All Clusters", "value": "All"}] +
                            [{"label": c, "value": c} for c in ["Value", "Premium", "Luxury"]],
                    value="All", clearable=False,
                    style={"backgroundColor": "#334155", "color": "#000"}),
            ]), style={"backgroundColor": "#1e293b", "border": "none"}), md=2),
        ], className="mb-3 g-2"),

        # Row 1: Map + SHAP
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("📍 Mumbai Ward Map", style={"color": "#f8fafc"}),
                dcc.Graph(id="map-chart", style={"height": "450px"}),
            ]), style={"backgroundColor": "#1e293b", "border": "none"}), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("🔍 SHAP Feature Importance", style={"color": "#f8fafc"}),
                dcc.Graph(id="shap-chart", style={"height": "450px"}),
            ]), style={"backgroundColor": "#1e293b", "border": "none"}), md=5),
        ], className="mb-3 g-2"),

        # Row 2: Scatter + Distribution
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("💰 Yield vs Price per SqFt", style={"color": "#f8fafc"}),
                dcc.Graph(id="scatter-chart", style={"height": "400px"}),
            ]), style={"backgroundColor": "#1e293b", "border": "none"}), md=7),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("📊 Cluster Distribution", style={"color": "#f8fafc"}),
                dcc.Graph(id="dist-chart", style={"height": "400px"}),
            ]), style={"backgroundColor": "#1e293b", "border": "none"}), md=5),
        ], className="mb-3 g-2"),

        # Row 3: Summary table
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("📋 Cluster Summary Statistics", style={"color": "#f8fafc"}),
            html.Div(id="summary-table"),
        ]), style={"backgroundColor": "#1e293b", "border": "none"}), md=12), className="mb-4 g-2"),

    ], fluid=True, style={"backgroundColor": "#0f172a", "minHeight": "100vh"})

    # ── Callbacks ─────────────────────────────────────────────────────────────
    @app.callback(
        [Output("map-chart", "figure"), Output("shap-chart", "figure"),
         Output("scatter-chart", "figure"), Output("dist-chart", "figure"),
         Output("summary-table", "children")],
        [Input("cluster-filter", "value")]
    )
    def update_charts(cluster):
        dff = df if cluster == "All" else df[df["Cluster_Label"] == cluster]

        # 1. Map
        map_fig = px.scatter_mapbox(
            dff, lat="Latitude", lon="Longitude", color="Cluster_Label",
            color_discrete_map=CLUSTER_COLORS, size="Sale_Price_INR",
            size_max=15, opacity=0.7, zoom=10, height=450,
            hover_data={"Ward": True, "Price_per_SqFt": ":.0f",
                        "Carpet_Area_SqFt": True, "Investment_Yield_Pct": ":.2f",
                        "Latitude": False, "Longitude": False, "Sale_Price_INR": False},
            mapbox_style="carto-darkmatter",
            center={"lat": 19.08, "lon": 72.88},
        )
        map_fig.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font_color="#f8fafc", margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.01),
        )

        # 2. SHAP chart
        if shap_df is not None:
            col = cluster if cluster in shap_df.columns else "Luxury"
            shap_sorted = shap_df[col].sort_values(ascending=True)
            shap_fig = go.Figure(go.Bar(
                x=shap_sorted.values, y=shap_sorted.index,
                orientation='h', marker_color="#8b5cf6",
                text=[f"{v:.3f}" for v in shap_sorted.values], textposition="outside",
            ))
            shap_fig.update_layout(
                title=f"SHAP Importance: {col}", title_font_size=13,
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                font_color="#f8fafc", margin=dict(l=10, r=10, t=35, b=10),
                xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
            )
        else:
            shap_fig = go.Figure()
            shap_fig.add_annotation(text="Run xai_shap.py first", showarrow=False,
                                     font=dict(color="#f8fafc", size=16))
            shap_fig.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b")

        # 3. Yield vs Price scatter
        scatter_fig = px.scatter(
            dff, x="Price_per_SqFt", y="Investment_Yield_Pct",
            color="Cluster_Label", color_discrete_map=CLUSTER_COLORS,
            opacity=0.6, hover_data=["Ward", "Carpet_Area_SqFt"],
        )
        scatter_fig.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font_color="#f8fafc", margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Price per SqFt (₹)", gridcolor="#334155"),
            yaxis=dict(title="Investment Yield (%)", gridcolor="#334155"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        # 4. Cluster distribution
        dist_data = df.groupby(["Cluster_Label", "Ward"]).size().reset_index(name="Count")
        dist_fig = px.bar(
            dist_data, x="Ward", y="Count", color="Cluster_Label",
            color_discrete_map=CLUSTER_COLORS, barmode="stack",
        )
        dist_fig.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font_color="#f8fafc", margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#334155", tickangle=-45),
            yaxis=dict(title="Properties", gridcolor="#334155"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        # 5. Summary table
        summary = dff.groupby("Cluster_Label").agg(
            Count=("Property_ID", "count"),
            Avg_Price_Cr=("Sale_Price_INR", lambda x: round(x.mean()/1e7, 2)),
            Avg_PSF=("Price_per_SqFt", lambda x: round(x.mean())),
            Avg_Area=("Carpet_Area_SqFt", lambda x: round(x.mean())),
            Avg_Yield=("Investment_Yield_Pct", lambda x: round(x.mean(), 2)),
            Avg_Metro_km=("Distance_to_Metro_km", lambda x: round(x.mean(), 2)),
        ).reset_index()

        table = dash_table.DataTable(
            data=summary.to_dict("records"),
            columns=[{"name": c, "id": c} for c in summary.columns],
            style_header={"backgroundColor": "#334155", "color": "#f8fafc",
                          "fontWeight": "bold", "border": "1px solid #475569"},
            style_cell={"backgroundColor": "#1e293b", "color": "#f8fafc",
                        "border": "1px solid #475569", "textAlign": "center",
                        "padding": "8px"},
            style_data_conditional=[
                {"if": {"filter_query": '{Cluster_Label} = "Luxury"'},
                 "backgroundColor": "#422006", "color": "#fbbf24"},
            ],
        )

        return map_fig, shap_fig, scatter_fig, dist_fig, table

    return app


if __name__ == "__main__":
    app = create_app()
    print("\n🚀 Dashboard running at http://localhost:8050")
    app.run(debug=True, port=8050)
