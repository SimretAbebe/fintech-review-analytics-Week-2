"""
Tailwind CSS UI styling helpers for the Streamlit dashboard.

Uses components.v1.html for rendered HTML blocks with Tailwind CSS.
"""

import streamlit.components.v1 as components

TAILWIND_HEAD = '<script src="https://cdn.tailwindcss.com"></script>'


def _render_html_block(body_html: str, height: int) -> None:
    """Wrap body HTML in a clean document with Tailwind CSS."""
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        {TAILWIND_HEAD}
    </head>
    <body style="margin:0; padding:0; background-color: transparent;">
        <div class="w-full h-full p-1 font-sans">
            {body_html}
        </div>
    </body>
    </html>
    """
    components.html(full_html, height=height, scrolling=False)


def render_header_banner(title: str, subtitle: str) -> None:
    """Render a clean black/white header banner with blue accent."""
    body = f"""
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-sm flex flex-col justify-center">
        <h1 class="text-2xl md:text-3xl font-extrabold text-white tracking-tight">
            {title}
        </h1>
        <p class="text-slate-400 text-sm mt-1">{subtitle}</p>
    </div>
    """
    _render_html_block(body, height=110)


def render_metric_cards(metrics: list[tuple[str, str]]) -> None:
    """Render a row of clean metric cards with high-contrast text and blue numbers."""
    cards_html = ""
    for label, value in metrics:
        cards_html += f"""
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm flex-1 min-w-[140px] flex flex-col justify-center">
            <p class="text-xs uppercase tracking-wider text-slate-400 font-semibold">{label}</p>
            <p class="text-3xl font-black text-blue-400 mt-1">{value}</p>
        </div>
        """
    body = f'<div class="flex gap-3 flex-wrap">{cards_html}</div>'
    _render_html_block(body, height=110)


def render_risk_verdict_card(is_high_risk: bool, probability: float) -> None:
    """Render a clean high-contrast risk verdict card."""
    if is_high_risk:
        border_color = "border-red-500/40"
        bg_color = "bg-red-950/20"
        text_color = "text-red-400"
        label = "HIGH RISK"
    else:
        border_color = "border-emerald-500/40"
        bg_color = "bg-emerald-950/20"
        text_color = "text-emerald-400"
        label = "NOT HIGH RISK"

    body = f"""
    <div class="bg-slate-900 border {border_color} {bg_color} rounded-xl p-6 text-center shadow-sm">
        <p class="{text_color} text-3xl font-black tracking-wider uppercase">{label}</p>
        <p class="text-slate-300 mt-2 text-sm">Predicted risk probability:
            <span class="font-bold text-white text-base ml-1">{probability:.1%}</span>
        </p>
    </div>
    """
    _render_html_block(body, height=140)


def render_language_badge(language_name: str) -> None:
    """Render a clean pill badge showing the detected language."""
    body = f"""
    <div class="inline-flex items-center gap-2 bg-blue-950/40 text-blue-400 border border-blue-500/30 text-xs font-semibold px-3 py-1.5 rounded-md">
        <span class="w-2 h-2 rounded-full bg-blue-500"></span>
        Detected language: <span class="text-white font-bold">{language_name}</span>
    </div>
    """
    _render_html_block(body, height=45)


def render_explanation_cards(explanations: list[str]) -> None:
    """Render SHAP explanation items as crisp slate cards with blue indicator borders."""
    cards_html = ""
    for explanation in explanations:
        cards_html += f"""
        <div class="bg-slate-900 border border-slate-800 border-l-4 border-l-blue-500 rounded-r-lg rounded-l-sm px-4 py-3 mb-2 shadow-sm">
            <p class="text-slate-200 text-sm font-medium">{explanation}</p>
        </div>
        """
    height = 50 + (len(explanations) * 55)
    _render_html_block(cards_html, height=height)