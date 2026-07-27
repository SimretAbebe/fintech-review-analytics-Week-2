"""
Tailwind CSS glassmorphism styling helpers for the Streamlit dashboard.

See tailwind_ui.py's earlier version history (removed) for why
components.v1.html is used instead of st.markdown for this - script
tags inserted via innerHTML do not execute in browsers, which is what
st.markdown(unsafe_allow_html=True) relies on internally.
"""

import streamlit.components.v1 as components

TAILWIND_HEAD = '<script src="https://cdn.tailwindcss.com"></script>'

# Shared dark gradient + glow-blob background used behind every glass
# panel, so all blocks feel visually consistent with each other.
_GLOW_BACKDROP = """
<div class="absolute inset-0 bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-900"></div>
<div class="absolute -top-10 -left-10 w-40 h-40 bg-cyan-500 rounded-full blur-3xl opacity-30"></div>
<div class="absolute -bottom-10 -right-10 w-40 h-40 bg-fuchsia-500 rounded-full blur-3xl opacity-30"></div>
"""


def _render_html_block(body_html: str, height: int) -> None:
    """Wrap body HTML in a full document (with Tailwind + glow backdrop) and render it."""
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        {TAILWIND_HEAD}
    </head>
    <body style="margin:0; padding:0;">
        <div class="relative w-full h-full overflow-hidden rounded-2xl">
            {_GLOW_BACKDROP}
            <div class="relative z-10 w-full h-full p-4">
                {body_html}
            </div>
        </div>
    </body>
    </html>
    """
    components.html(full_html, height=height, scrolling=False)


def render_header_banner(title: str, subtitle: str) -> None:
    """Render a glassmorphism header banner."""
    body = f"""
    <div class="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6 shadow-2xl">
        <h1 class="text-3xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-cyan-300 via-sky-200 to-fuchsia-300 mb-1">
            {title}
        </h1>
        <p class="text-slate-300 text-sm">{subtitle}</p>
    </div>
    """
    _render_html_block(body, height=150)


def render_metric_cards(metrics: list[tuple[str, str]]) -> None:
    """Render a row of glassmorphism metric cards."""
    cards_html = ""
    for label, value in metrics:
        cards_html += f"""
        <div class="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-4 shadow-xl flex-1 min-w-[140px]">
            <p class="text-xs uppercase tracking-wide text-slate-300 font-medium">{label}</p>
            <p class="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 to-fuchsia-300 mt-1">{value}</p>
        </div>
        """
    body = f'<div class="flex gap-3 flex-wrap">{cards_html}</div>'
    _render_html_block(body, height=130)


def render_risk_verdict_card(is_high_risk: bool, probability: float) -> None:
    """Render a large glassmorphism risk verdict card."""
    if is_high_risk:
        glow = "bg-rose-500"
        text_color = "text-rose-300"
        icon = ""
        label = "HIGH RISK"
    else:
        glow = "bg-emerald-500"
        text_color = "text-emerald-300"
        icon = ""
        label = "NOT HIGH RISK"

    body = f"""
    <div class="relative">
        <div class="absolute inset-0 {glow} rounded-2xl blur-2xl opacity-20"></div>
        <div class="relative bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6 text-center shadow-2xl">
            <p class="text-5xl mb-2">{icon}</p>
            <p class="{text_color} text-3xl font-extrabold tracking-wide">{label}</p>
            <p class="text-slate-300 mt-2">Predicted risk probability:
                <span class="font-bold text-white">{probability:.1%}</span>
            </p>
        </div>
    </div>
    """
    _render_html_block(body, height=210)


def render_language_badge(language_name: str) -> None:
    """Render a glassmorphism badge showing the detected language."""
    colors = {
        "English": "from-sky-400 to-blue-500",
        "Amharic": "from-amber-400 to-orange-500",
        "Afaan Oromo": "from-fuchsia-400 to-purple-500",
    }
    gradient = colors.get(language_name, "from-slate-400 to-slate-500")
    body = f"""
    <span class="inline-block bg-gradient-to-r {gradient} text-white text-sm font-semibold px-4 py-2 rounded-full shadow-lg">
        Detected language: {language_name}
    </span>
    """
    _render_html_block(body, height=55)


def render_explanation_cards(explanations: list[str]) -> None:
    """Render SHAP explanation bullet points as glassmorphism cards."""
    cards_html = ""
    for explanation in explanations:
        cards_html += f"""
        <div class="bg-white/10 backdrop-blur-xl border-l-4 border-cyan-400 border border-white/20 rounded-r-xl rounded-l-md px-4 py-3 mb-2 shadow-lg">
            <p class="text-slate-100 text-sm">{explanation}</p>
        </div>
        """
    height = 70 + (len(explanations) * 60)
    _render_html_block(cards_html, height=height)