import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import xgboost as xgb
import shap
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import resample
from flask_caching import Cache
import os

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
# Use relative paths (assumes the "student" folder is in the project directory)
por_path = "student/student-por.csv"
mat_path = "student/student-mat.csv"
df_por = pd.read_csv(por_path, sep=";")
df_math = pd.read_csv(mat_path, sep=";")

common_columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                  'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian']
df_merged = pd.merge(df_math, df_por, on=common_columns, suffixes=('_mat', '_por'))
df_merged['G3_avg'] = df_merged[['G3_mat', 'G3_por']].mean(axis=1)
df_merged['G1_avg'] = df_merged[['G1_mat', 'G1_por']].mean(axis=1)
df_merged['studytime_avg'] = df_merged[['studytime_mat', 'studytime_por']].mean(axis=1)
df_merged['absences_avg'] = df_merged[['absences_mat', 'absences_por']].mean(axis=1)

# Filter options
school_options = [{'label': s, 'value': s} for s in sorted(df_merged['school'].unique())]
school_options.insert(0, {'label': 'All', 'value': 'All'})
studytime_vals = sorted(set(df_merged['studytime_mat'].unique()).union(set(df_merged['studytime_por'].unique())))
studytime_options = [{'label': str(st), 'value': st} for st in studytime_vals]
studytime_options.insert(0, {'label': 'All', 'value': 'All'})
subject_options = [
    {'label': 'Math', 'value': 'mat'},
    {'label': 'Portuguese', 'value': 'por'},
    {'label': 'Average', 'value': 'avg'}
]
gender_options = [{'label': g, 'value': g} for g in sorted(df_merged['sex'].unique())]
gender_options.insert(0, {'label': 'All', 'value': 'All'})

# -------------------------------
# AI Model Integration & Fairness Calculations
# -------------------------------
# For demonstration, we will use a simplified feature set and train an XGBoost regressor
features = ['G1_avg', 'studytime_avg', 'absences_avg', 'age']
X = df_merged[features]
y = df_merged['G3_avg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)

# Compute fairness metric: demographic parity difference by gender based on predictions
df_merged['predicted_G3'] = model.predict(df_merged[features])
female_avg = df_merged[df_merged['sex'] == 'F']['predicted_G3'].mean()
male_avg = df_merged[df_merged['sex'] == 'M']['predicted_G3'].mean()
demographic_parity_diff = abs(female_avg - male_avg)

# SHAP explainability (global summary)
explainer = shap.Explainer(model)
shap_values = explainer(X_train)
# For simplicity, we create a global SHAP summary using Plotly Express (this is an approximation)
shap_mean_abs = [abs(shap_values[:, i]).mean() for i in range(len(features))]
shap_df = pd.DataFrame({"Feature": features, "Mean Absolute SHAP Value": shap_mean_abs})
shap_fig = px.bar(shap_df, x="Feature", y="Mean Absolute SHAP Value",
                  title="Global Feature Importance (SHAP)")

# LIME explainer (for a sample instance)
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=features, mode='regression'
)
# Pre-calculate a LIME explanation for the first instance of X_test
lime_exp = lime_explainer.explain_instance(X_test.iloc[0].values, model.predict, num_features=4)
lime_fig = px.bar(x=[f for f, w in lime_exp.as_list()],
                  y=[w for f, w in lime_exp.as_list()],
                  labels={"x": "Feature", "y": "Contribution"},
                  title="Local Explanation (LIME) for a Sample Prediction")

# -------------------------------
# Caching Setup (using SimpleCache for demonstration)
# -------------------------------
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache'})

# -------------------------------
# Dash App Initialization
# -------------------------------
external_stylesheets = [dbc.themes.LUX]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# -------------------------------
# Dashboard Layout with Additional Tabs for Fairness and Model Insights
# -------------------------------
app.layout = dbc.Container([
    dcc.Store(id="dark-mode-store", data=False),  # Store for dark mode

    dbc.NavbarSimple(
        children=[
            dbc.Switch(
                id="dark-mode-switch",
                label="Dark Mode",
                value=False,
                style={"marginLeft": "20px", "color": "white"}
            ),
            dbc.Button("Filters", id="open-offcanvas", color="secondary", outline=True, style={"marginLeft": "20px"})
        ],
        brand="Student Performance Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        fixed="top",
        style={"boxShadow": "0 4px 12px rgba(0,0,0,0.2)"}
    ),

    html.Br(), html.Br(), html.Br(),

    dbc.Offcanvas(
        id="offcanvas-filters",
        title="Filter Options",
        is_open=False,
        placement="start",
        children=[
            html.Div([
                dbc.Label("Select Subject", className="fw-bold"),
                dcc.Dropdown(
                    id="subject-dropdown",
                    options=subject_options,
                    value="avg",
                    clearable=False,
                    className="custom-dropdown",
                    style={"borderRadius": "5px"}
                )
            ], className="mb-3"),
            html.Div([
                dbc.Label("Select Gender", className="fw-bold"),
                dcc.Dropdown(
                    id="gender-dropdown",
                    options=gender_options,
                    value="All",
                    clearable=False,
                    className="custom-dropdown",
                    style={"borderRadius": "5px"}
                )
            ], className="mb-3"),
            html.Div([
                dbc.Label("Select School", className="fw-bold"),
                dcc.Dropdown(
                    id="school-dropdown",
                    options=school_options,
                    value="All",
                    clearable=False,
                    className="custom-dropdown",
                    style={"borderRadius": "5px"}
                )
            ], className="mb-3"),
            html.Div([
                dbc.Label("Select Study Time", className="fw-bold"),
                dcc.Dropdown(
                    id="studytime-dropdown",
                    options=studytime_options,
                    value="All",
                    clearable=False,
                    className="custom-dropdown",
                    style={"borderRadius": "5px"}
                )
            ], className="mb-3"),
            html.Div([
                dbc.Label("Select Age Range", className="fw-bold"),
                dcc.RangeSlider(
                    id="age-slider",
                    min=df_merged["age"].min(),
                    max=df_merged["age"].max(),
                    value=[df_merged["age"].min(), df_merged["age"].max()],
                    marks={str(age): str(age) for age in range(df_merged["age"].min(), df_merged["age"].max() + 1)}
                )
            ])
        ],
        style={"width": "300px", "padding": "20px"}
    ),

    dbc.Tabs(
        id="tabs",
        active_tab="tab-overview",
        children=[
            dbc.Tab(label="Overview", tab_id="tab-overview"),
            dbc.Tab(label="Histogram", tab_id="tab-hist"),
            dbc.Tab(label="Scatter Plot", tab_id="tab-scatter"),
            dbc.Tab(label="Box Plot", tab_id="tab-box"),
            dbc.Tab(label="Data Table", tab_id="tab-table"),
            dbc.Tab(label="Fairness Metrics", tab_id="tab-fairness"),
            dbc.Tab(label="Model Insights", tab_id="tab-model")
        ],
        className="mb-4"
    ),

    dcc.Loading(
        id="loading-indicator",
        type="circle",
        children=html.Div(id="tab-content", style={"marginBottom": "40px"})
    ),

    dbc.Row([
        dbc.Col(
            html.Span("© 2025 Student Performance Dashboard", id="footer-text", className="footer-text"),
            width=12
        )
    ], className="mt-4")
],
    fluid=True,
    id="main-container",
    style={"paddingTop": "100px", "backgroundColor": "#F8F9FA", "minHeight": "100vh"}
)

# -------------------------------
# Callback for Dark Mode Toggle (updates container, footer, and tabs class)
# -------------------------------
@app.callback(
    Output("main-container", "style"),
    Output("dark-mode-store", "data"),
    Output("footer-text", "style"),
    Output("tabs", "className"),
    Input("dark-mode-switch", "value"),
    State("main-container", "style"),
    prevent_initial_call=True
)
def update_dark_mode(is_dark, current_style):
    new_style = current_style.copy() if current_style else {}
    if is_dark:
        new_style.update({"backgroundColor": "#2B2B2B", "color": "#FFFFFF"})
        footer_style = {"color": "#FFFFFF"}
        tabs_class = "tabs-dark"
    else:
        new_style.update({"backgroundColor": "#F8F9FA", "color": "#000000"})
        footer_style = {"color": "#000000"}
        tabs_class = "tabs-light"
    return new_style, is_dark, footer_style, tabs_class

# -------------------------------
# Callback to Update Offcanvas Filter Menu Class Based on Dark Mode
# -------------------------------
@app.callback(
    Output("offcanvas-filters", "className"),
    Input("dark-mode-store", "data")
)
def update_offcanvas_class(dark_mode):
    return "dark-offcanvas" if dark_mode else ""

# -------------------------------
# Callback to Update Offcanvas Filter Menu Style Based on Dark Mode
# -------------------------------
@app.callback(
    Output("offcanvas-filters", "style"),
    Input("dark-mode-store", "data")
)
def update_offcanvas_style(dark_mode):
    base_style = {"width": "300px", "padding": "20px"}
    if dark_mode:
        base_style.update({"backgroundColor": "#2B2B2B", "color": "#FFFFFF"})
    else:
        base_style.update({"backgroundColor": "#FFFFFF", "color": "#000000"})
    return base_style

# -------------------------------
# Callback to Toggle Offcanvas Filters
# -------------------------------
@app.callback(
    Output("offcanvas-filters", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    State("offcanvas-filters", "is_open")
)
def toggle_offcanvas(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# -------------------------------
# Callback to Render Tab Content
# -------------------------------
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"),
     Input("subject-dropdown", "value"),
     Input("gender-dropdown", "value"),
     Input("school-dropdown", "value"),
     Input("studytime-dropdown", "value"),
     Input("age-slider", "value"),
     Input("dark-mode-store", "data")]
)
def render_tab_content(active_tab, selected_subject, selected_gender, selected_school, selected_studytime, age_range,
                       dark_mode):
    filtered_df = df_merged.copy()
    if selected_gender != "All":
        filtered_df = filtered_df[filtered_df["sex"] == selected_gender]
    if selected_school != "All":
        filtered_df = filtered_df[filtered_df["school"] == selected_school]
    if selected_studytime != "All":
        if selected_subject == "mat":
            filtered_df = filtered_df[filtered_df["studytime_mat"] == selected_studytime]
        elif selected_subject == "por":
            filtered_df = filtered_df[filtered_df["studytime_por"] == selected_studytime]
        else:
            filtered_df = filtered_df[filtered_df["studytime_mat"] == selected_studytime]
    filtered_df = filtered_df[(filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])]

    if selected_subject == "mat":
        grade_col = "G3_mat"
        g1_col = "G1_mat"
    elif selected_subject == "por":
        grade_col = "G3_por"
        g1_col = "G1_por"
    else:
        grade_col = "G3_avg"
        g1_col = "G1_avg"

    template = "plotly_dark" if dark_mode else "plotly_white"

    # Overview Tab: Cards and summary table
    if active_tab == "tab-overview":
        card1 = dbc.Card(
            dbc.CardBody([
                html.H5("Avg Final Grade", className="card-title"),
                html.H2(f"{filtered_df[grade_col].mean():.1f}", className="card-text")
            ]),
            color="info", inverse=True,
            style={"boxShadow": "0 4px 12px rgba(0,0,0,0.25)", "borderRadius": "10px"}
        )
        card2 = dbc.Card(
            dbc.CardBody([
                html.H5("Avg First Period Grade", className="card-title"),
                html.H2(f"{filtered_df[g1_col].mean():.1f}", className="card-text")
            ]),
            color="success", inverse=True,
            style={"boxShadow": "0 4px 12px rgba(0,0,0,0.25)", "borderRadius": "10px"}
        )
        card3 = dbc.Card(
            dbc.CardBody([
                html.H5("Avg Age", className="card-title"),
                html.H2(f"{filtered_df['age'].mean():.1f}", className="card-text")
            ]),
            color="warning", inverse=True,
            style={"boxShadow": "0 4px 12px rgba(0,0,0,0.25)", "borderRadius": "10px"}
        )
        cards_row = dbc.Row([
            dbc.Col(card1, width=4),
            dbc.Col(card2, width=4),
            dbc.Col(card3, width=4)
        ], className="mb-4")
        summary_df = filtered_df[[grade_col, g1_col, "age", "sex", "school"]].describe().transpose().reset_index()
        if dark_mode:
            header_bg = "#2B2B2B"
            header_color = "#FFFFFF"
            cell_bg = "#3B3B3B"
            cell_color = "#FFFFFF"
        else:
            header_bg = "#F8F9FA"
            header_color = "#000000"
            cell_bg = "#FFFFFF"
            cell_color = "#000000"
        data_table = dash_table.DataTable(
            id="summary-table",
            columns=[{"name": i, "id": i} for i in summary_df.columns],
            data=summary_df.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": header_bg, "fontWeight": "bold", "color": header_color},
            style_cell={"textAlign": "center", "padding": "5px", "backgroundColor": cell_bg, "color": cell_color},
            page_size=10
        )
        return dbc.Container([cards_row, dbc.Row([dbc.Col(data_table, width=12)])])

    # Histogram Tab
    if active_tab == "tab-hist":
        hist_fig = px.histogram(
            filtered_df,
            x=grade_col,
            nbins=20,
            title="Final Grade Distribution",
            color="sex",
            color_discrete_map={"F": "#FF6F61", "M": "#2E86C1"}
        )
        hist_fig.update_layout(
            bargap=0.1,
            xaxis_title="Final Grade",
            yaxis_title="Count",
            template=template,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        return dcc.Graph(figure=hist_fig, config={"displayModeBar": False})

    # Scatter Plot Tab
    if active_tab == "tab-scatter":
        scatter_fig = px.scatter(
            filtered_df,
            x=g1_col,
            y=grade_col,
            title="Scatter Plot: First Period vs Final Grade",
            color="sex",
            color_discrete_map={"F": "#FF6F61", "M": "#2E86C1"},
            hover_data=["age", "studytime_avg", "absences_avg"]
        )
        scatter_fig.update_layout(
            xaxis_title="First Period Grade",
            yaxis_title="Final Grade",
            template=template,
            margin=dict(l=40, r=40, t=50, b=50)
        )
        return dcc.Graph(figure=scatter_fig, config={"displayModeBar": False})

    # Box Plot Tab
    if active_tab == "tab-box":
        box_fig = px.box(
            filtered_df,
            x="sex",
            y=grade_col,
            title="Box Plot: Final Grade by Gender",
            color="sex",
            color_discrete_map={"F": "#FF6F61", "M": "#2E86C1"}
        )
        box_fig.update_layout(
            xaxis_title="Gender",
            yaxis_title="Final Grade",
            template=template,
            margin=dict(l=40, r=40, t=50, b=50)
        )
        return dcc.Graph(figure=box_fig, config={"displayModeBar": False})

    # Data Table Tab
    if active_tab == "tab-table":
        display_cols = common_columns + [grade_col, g1_col, "age", "sex", "studytime_avg", "absences_avg"]
        if dark_mode:
            header_bg = "#2B2B2B"
            header_color = "#FFFFFF"
            cell_bg = "#3B3B3B"
            cell_color = "#FFFFFF"
        else:
            header_bg = "#F8F9FA"
            header_color = "#000000"
            cell_bg = "#FFFFFF"
            cell_color = "#000000"
        table = dash_table.DataTable(
            id="data-table",
            columns=[{"name": col, "id": col} for col in display_cols],
            data=filtered_df[display_cols].to_dict("records"),
            style_table={"overflowX": "auto", "minWidth": "100%"},
            style_header={"backgroundColor": header_bg, "fontWeight": "bold", "color": header_color},
            style_cell={"textAlign": "center", "padding": "5px", "backgroundColor": cell_bg, "color": cell_color},
            page_size=15
        )
        return dbc.Container([html.H4("Filtered Data", className="text-center mb-3",
                                       style={"color": "#FFFFFF" if dark_mode else "#000000"}), table])

    # Fairness Metrics Tab
    if active_tab == "tab-fairness":
        fairness_metrics = html.Div([
            html.H4("Fairness Metrics", className="text-center mb-3", style={"color": "#FFFFFF" if dark_mode else "#000000"}),
            html.P(f"Demographic Parity Difference (|Female Avg - Male Avg|): {demographic_parity_diff:.2f}",
                   style={"color": "#FFFFFF" if dark_mode else "#000000", "textAlign": "center"}),
            html.P(f"XGBoost Model R² Score: {r2:.2f}",
                   style={"color": "#FFFFFF" if dark_mode else "#000000", "textAlign": "center"})
        ])
        return dbc.Container(fairness_metrics)

    # Model Insights Tab
    if active_tab == "tab-model":
        model_insights = html.Div([
            html.H4("Model Insights", className="text-center mb-3", style={"color": "#FFFFFF" if dark_mode else "#000000"}),
            html.P(f"XGBoost Model Performance (R² Score): {r2:.2f}",
                   style={"color": "#FFFFFF" if dark_mode else "#000000", "textAlign": "center"}),
            dcc.Graph(figure=shap_fig, config={"displayModeBar": False}),
            dcc.Graph(figure=lime_fig, config={"displayModeBar": False}),
            html.P("Click on a student record (in the Data Table) for a detailed local explanation (LIME) in the future.",
                   style={"color": "#FFFFFF" if dark_mode else "#000000", "textAlign": "center"})
        ])
        return dbc.Container(model_insights)

    return html.Div("No tab selected", className="text-center")

# -------------------------------
# Run the App
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)