"""MiniGPT - Streamlit interface for the from-scratch GPT demo.

This module only renders the UI; model loading and generation live in
``src.inference`` and are imported from there.
"""

from __future__ import annotations

from collections.abc import Iterator

import streamlit as st

from src.config import PROJECT_ROOT, GenerationConfig
from src.inference import LoadedModel, load_for_inference, load_metrics, stream_completion

STYLES_PATH = PROJECT_ROOT / "assets" / "styles.css"
DEFAULTS = GenerationConfig()

EXAMPLES: dict[str, str] = {
    "Story opening": "Once upon a time",
    "Little robot": "The little robot",
    "Lily and Tom": "Lily and Tom went to",
}


@st.cache_resource
def get_model() -> LoadedModel:
    """Load the trained model once and cache it across reruns."""
    return load_for_inference()


def _metric_float(metrics: dict[str, object], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _metric_int(metrics: dict[str, object], key: str, default: int = 0) -> int:
    value = metrics.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


@st.cache_data
def get_metrics() -> dict[str, object]:
    return load_metrics()


def _inject_css() -> None:
    st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _render_sidebar(metrics: dict[str, object]) -> None:
    st.sidebar.markdown("## About the model")
    st.sidebar.markdown(
        "A **GPT-style decoder-only transformer** built from scratch in PyTorch, "
        "with causal multi-head attention and autoregressive sampling."
    )
    st.sidebar.markdown("---")
    val_ppl = _metric_float(metrics, "val_perplexity")
    num_params = _metric_int(metrics, "num_params")
    st.sidebar.markdown(
        f'<div class="stat-box"><div class="stat-value">{val_ppl:.2f}</div>'
        '<div class="stat-label">Validation perplexity</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("")
    cols = st.sidebar.columns(2)
    cols[0].markdown(
        f'<div class="stat-box"><div class="stat-value">{num_params / 1e6:.1f}M</div>'
        '<div class="stat-label">Parameters</div></div>',
        unsafe_allow_html=True,
    )
    dataset = metrics.get("dataset", "unknown")
    tokenizer_type = metrics.get("tokenizer_type", "unknown")
    cols[1].markdown(
        f'<div class="stat-box"><div class="stat-value">{tokenizer_type!s}</div>'
        '<div class="stat-label">Tokenizer</div></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("")
    st.sidebar.caption(f"Trained on {dataset} · {tokenizer_type} tokenizer")


def main() -> None:
    st.set_page_config(page_title="MiniGPT", page_icon="🧠", layout="centered")
    _inject_css()
    metrics = get_metrics()
    loaded = get_model()

    st.markdown('<div class="main-title">MiniGPT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A GPT-style language model built from scratch in PyTorch — '
        "tokens stream live as they are generated.</div>",
        unsafe_allow_html=True,
    )
    _render_sidebar(metrics)

    example = st.selectbox("Example prompt", ["Custom", *list(EXAMPLES.keys())])
    default_prompt = "" if example == "Custom" else EXAMPLES[example]
    prompt = st.text_area("Prompt", value=default_prompt, height=100)

    col1, col2 = st.columns(2)
    max_new_tokens = col1.slider(
        "Max new tokens", min_value=1, max_value=512, value=DEFAULTS.max_new_tokens
    )
    temperature = col2.slider(
        "Temperature", min_value=0.0, max_value=2.0, value=DEFAULTS.temperature, step=0.05
    )
    col3, col4 = st.columns(2)
    top_k = col3.slider("Top-k (0 = off)", min_value=0, max_value=200, value=DEFAULTS.top_k or 0)
    top_p = col4.slider(
        "Top-p", min_value=0.0, max_value=1.0, value=DEFAULTS.top_p or 0.95, step=0.01
    )

    if st.button("Generate", type="primary", disabled=not prompt.strip()):
        st.markdown("**Generated text**")
        output = st.empty()
        full_text = prompt

        def _stream() -> Iterator[str]:
            nonlocal full_text
            for chunk in stream_completion(
                loaded,
                prompt,
                max_new_tokens,
                temperature,
                top_k if top_k > 0 else None,
                top_p,
            ):
                full_text += chunk
                output.markdown(
                    f'<div class="output-box">{full_text}</div>',
                    unsafe_allow_html=True,
                )
                yield chunk

        st.write_stream(_stream)


if __name__ == "__main__":
    main()
