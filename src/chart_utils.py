"""
Plotly chart helpers for the dashboard.

Why Plotly instead of st.bar_chart:
    Streamlit's built-in st.bar_chart auto-rotates long category labels
    vertically when they don't fit, with no way to override this.
    Plotly gives full control over label angle (tickangle=0 forces
    horizontal text) and lets us match the dashboard's dark,
    glassmorphism color scheme.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DARK_BG = "rgba(0,0,0,0)"  # transparent, so Streamlit's own dark theme shows through
GRID_COLOR = "rgba(255,255,255,0.08)"
TEXT_COLOR = "#f8fafc"
BAR_COLORS = ["#3b82f6", "#2563eb", "#60a5fa", "#1d4ed8", "#93c5fd"]


def render_value_counts_bar_chart(value_counts: pd.Series, title: str) -> None:
    """Render a horizontal-label bar chart from a pandas value_counts() Series."""
    labels = value_counts.index.astype(str).tolist()
    values = value_counts.values.tolist()
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(labels))]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors, line=dict(width=0)),
                text=values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(
            tickangle=0,  # force horizontal labels
            gridcolor=GRID_COLOR,
            tickfont=dict(size=11),
        ),
        yaxis=dict(gridcolor=GRID_COLOR),
        margin=dict(t=50, b=80, l=40, r=20),
        height=380,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)