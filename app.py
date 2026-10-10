"""AI Music Generator - Streamlit interface.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import math
from dataclasses import replace
from pathlib import Path

import streamlit as st

from src import config
from src.audio_utils import midi_to_wav_bytes
from src.generate import generate_music, load_generation_assets, model_files_status
from src.history import add_entry, clear_history, load_history, untracked_midi_files
from src.midi_utils import find_midi_files
from src.preprocessing import preprocess_dataset, scan_dataset
from src.train import train_model
from src.utils import MusicGenError, ValidationError, load_json, setup_logging

setup_logging()
logger = logging.getLogger("src.app")

st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="wide")

NAV = [
    ("Dashboard", "🏠"),
    ("Dataset", "🗄️"),
    ("Training", "⚙️"),
    ("Generate Music", "🎵"),
    ("History", "🕘"),
    ("About", "ℹ️"),
]
PAGES = [name for name, _ in NAV]


SETTINGS = [
    ("tr_seq", "Sequence Length", 10, 150, 5),
    ("tr_batch", "Batch Size", 8, 128, 8),
    ("tr_epochs", "Epochs", 1, 50, 1),
]
MAX_SEED = 2**31 - 1

CSS = """
<style>
:root { --bg:#040a1c; --panel:rgba(11,22,56,.78); --border:rgba(96,130,255,.20); --muted:#9db0d8;
        --blue:#3b82f6; --violet:#8b5cf6; --cyan:#22d3ee; --green:#34d399; }
.stApp { background: radial-gradient(1100px 520px at 70% -8%, rgba(59,60,200,.28), transparent 60%),
                     radial-gradient(800px 500px at 0% 0%, rgba(20,60,160,.22), transparent 55%), var(--bg); }
[data-testid="stHeader"] { background: rgba(4,10,28,.92); border-bottom: 1px solid var(--border); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#060f2b,#050b20); border-right: 1px solid var(--border); }
.block-container { padding-top: 2.4rem; max-width: 1250px; }
[data-testid="stAppDeployButton"] button, .stAppDeployButton button {
    background: linear-gradient(90deg,#3b6cff,#7c4dff); color:#fff; border:0; border-radius:10px; }

@keyframes fadeIn { from {opacity:0; transform:translateY(6px);} to {opacity:1; transform:none;} }
@keyframes pulse { 0%,100% {transform:scaleY(1);} 50% {transform:scaleY(.8);} }

/* ---------- sidebar ---------- */
.brand { display:flex; align-items:center; gap:10px; font-size:1.3rem; font-weight:700; margin:2px 0 16px; }
.brand .logo { font-size:1.6rem; }
.side-title { display:flex; align-items:center; gap:8px; color:#cfe0ff; font-weight:600; margin:12px 0 8px; }
[data-testid="stSidebar"] .stButton > button { width:100%; justify-content:flex-start; text-align:left; background:transparent;
    color:#dbe6ff; border:1px solid transparent; border-radius:12px; padding:.55rem .85rem; font-weight:600; box-shadow:none; }
[data-testid="stSidebar"] .stButton > button p { text-align:left; }
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(59,130,246,.14); border-color: var(--border); transform:none; }
[data-testid="stSidebar"] button[kind="primary"], [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, rgba(59,100,255,.45), rgba(100,70,210,.35)) !important;
    border-color: rgba(110,140,255,.55) !important; box-shadow: inset 3px 0 0 #4f8bff !important; min-height:auto; font-size:1rem; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button { justify-content:center; padding:.35rem 0; }
.set-label { color:#e8efff; font-weight:600; margin:10px 0 0; }
.set-value { font-size:1.2rem; font-weight:600; padding-top:4px; }
.value-row { display:flex; justify-content:space-between; align-items:baseline; font-weight:600; margin:8px 0 2px; }
.hint { color:var(--muted); font-size:.9rem; margin:2px 0 14px; line-height:1.5; }
.promo { display:flex; align-items:center; gap:14px; padding:18px; margin-top:18px; border-radius:16px; border:1px solid var(--border);
         background: linear-gradient(135deg, rgba(40,80,220,.35), rgba(124,58,237,.35)); }
.promo b { color:#7dd3fc; display:block; font-size:1.05rem; } .promo span { color:var(--muted); font-size:.85rem; }
.mini { display:flex; gap:3px; align-items:center; height:40px; }
.mini i { width:4px; border-radius:2px; background:linear-gradient(#22d3ee,#8b5cf6); }

/* ---------- hero ---------- */
.hero { position:relative; overflow:hidden; display:flex; flex-wrap:wrap; gap:20px; align-items:center; justify-content:space-between;
        padding:28px 34px; border-radius:22px; border:1px solid rgba(96,130,255,.35); animation:fadeIn .5s ease both;
        background: linear-gradient(120deg,#07163f 0%, #0b2a7a 45%, #3b1a8f 75%, #7c1fb8 100%); }
.hero .badge { display:inline-block; padding:5px 16px; border-radius:999px; font-size:.85rem; font-weight:600;
               border:1px solid rgba(125,211,252,.6); background:rgba(5,15,50,.6); color:#fff; margin-bottom:10px; }
.hero h1 { margin:0 0 6px; font-size:3.1rem; line-height:1.1; font-weight:800; letter-spacing:-.5px;
           background: linear-gradient(90deg,#22d3ee,#60a5fa 50%,#a78bfa); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p { margin:.55rem 0 0; color:#e3eaff; font-size:1.02rem; }
.viz { display:flex; align-items:center; gap:3px; height:150px; }
.viz i { width:5px; border-radius:3px; animation:pulse 2.6s ease-in-out infinite; }

/* ---------- status, cards, panels ---------- */
.status-row { display:flex; flex-wrap:wrap; gap:12px; margin:16px 0 8px; }
.pill { padding:8px 18px; border-radius:999px; font-size:.92rem; font-weight:600; border:1px solid var(--border); background:rgba(8,18,48,.85); }
.pill.ready { color:var(--green); } .pill.notready { color:#fcd34d; } .pill.training { color:#67e8f9; } .pill.error { color:#fca5a5; }
.panel-title { display:flex; align-items:center; gap:10px; font-size:1.55rem; font-weight:700; margin:0 0 .6rem; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:14px; margin:12px 0; }
.card { padding:16px 18px; border-radius:16px; background:var(--panel); border:1px solid var(--border); animation:fadeIn .5s ease both; overflow-wrap:anywhere; }
.card .label { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }
.card .value { font-size:1.45rem; font-weight:700; margin-top:4px; } .card .sub { color:var(--muted); font-size:.82rem; margin-top:2px; }
.steps { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin:14px 0; }
.step { padding:12px 14px; border-radius:14px; border:1px solid var(--border); background:var(--panel); font-size:.9rem; }
.step.done { border-color:#22c55e77; } .step.current { border-color:var(--blue); box-shadow:0 0 0 1px var(--blue) inset; }
.empty { padding:22px; border-radius:16px; border:1px dashed var(--border); color:var(--muted); text-align:center; }
.track { padding:12px 16px; border-radius:14px; background:var(--panel); border:1px solid var(--border); overflow-wrap:anywhere; }
.track .t-name { font-weight:600; } .track .t-meta { color:var(--muted); font-size:.82rem; margin-top:2px; }

/* ---------- player ---------- */
.player-head { display:flex; gap:14px; align-items:center; }
.album { width:66px; height:66px; flex:none; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:2rem;
         background: linear-gradient(135deg,#8b5cf6,#d946ef); }
.p-title { font-weight:700; font-size:1.05rem; } .p-meta { color:var(--muted); font-size:.9rem; overflow-wrap:anywhere; }
[data-testid="stAudio"] audio, audio { width:100%; color-scheme:dark; border-radius:14px; }

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button { border-radius:12px; font-weight:600; color:#e6eeff; background:rgba(15,30,70,.9);
    border:1px solid var(--border); transition:transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 6px 18px rgba(59,100,255,.35); border-color:rgba(130,160,255,.6); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    width:100%; min-height:3.3rem; font-size:1.08rem; color:#fff; border:0; background:linear-gradient(90deg,#2b7bff,#7c4dff); }
.stButton > button:disabled { opacity:.45; }
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
    background: rgba(8,16,42,.9) !important; border-radius:12px !important; }
[data-testid="stSliderThumbValue"], [data-testid="stTickBar"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { display:none; }
@media (max-width: 900px) { .viz { display:none; } .hero h1 { font-size:2.3rem; } }
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(value: object) -> str:
    return html.escape(str(value))


def card(label: str, value: object, sub: str = "") -> str:
    sub_html = f'<div class="sub">{esc(sub)}</div>' if sub else ""
    return f'<div class="card"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>{sub_html}</div>'


def show_cards(cards: list[str]) -> None:
    st.markdown('<div class="grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def empty_state(text: str) -> None:
    st.markdown(f'<div class="empty">{esc(text).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


def panel_title(icon: str, text: str) -> None:
    st.markdown(f'<div class="panel-title">{icon} {esc(text)}</div>', unsafe_allow_html=True)


def value_row(label: str, value: object) -> None:
    st.markdown(f'<div class="value-row"><span>{esc(label)}</span><span>{esc(value)}</span></div>', unsafe_allow_html=True)


def show_error(title: str, exc: Exception) -> None:
    """Friendly error box: what / why / how. Technical details stay in the logs."""
    if isinstance(exc, MusicGenError):
        body = f"**Reason:** {exc.reason or exc.message}\n\n**How to fix:** {exc.fix or 'See logs/ for details.'}"
        logger.error("%s: %s", title, exc.message)
    else:
        logger.exception("%s", title)
        body = "**Reason:** An unexpected problem occurred.\n\n**How to fix:** Try again; if it continues, check the log file in logs/."
    st.error(f"❌ {title}\n\n{body}")


def record_history(kind: str, details: dict) -> None:
    """Save a history entry; a history problem must never break the main action."""
    try:
        add_entry(kind, details)
    except Exception as exc:  
        logger.warning("Could not save history entry: %s", exc)


def read_json(path: Path) -> dict | None:
    try:
        data = load_json(path) if path.exists() else None
    except MusicGenError:
        return None
    return data if isinstance(data, dict) else None


def bars_html(count: int, max_height: int, css_class: str, spread: float = 0.22) -> str:
    """Decorative waveform bars (pure HTML/CSS, blue -> violet -> magenta)."""
    parts = []
    for i in range(count):
        x = i / max(count - 1, 1)
        envelope = math.exp(-(((x - 0.5) / spread) ** 2))
        height = 8 + max_height * envelope * (0.55 + 0.45 * abs(math.sin(i * 0.9))) + 6 * abs(math.sin(i * 2.3))
        hue = 200 + 100 * x
        parts.append(
            f'<i style="height:{height:.0f}px;background:hsl({hue:.0f},95%,62%);'
            f'box-shadow:0 0 8px hsl({hue:.0f},95%,60%);animation-delay:{i * 0.05:.2f}s"></i>'
        )
    return f'<div class="{css_class}" aria-hidden="true">{"".join(parts)}</div>'


HERO_HTML = (
    '<div class="hero"><div>'
    '<span class="badge">CodeAlpha</span>'
    "<h1>AI Music Generator</h1>"
    "<p>Create original melodies with an LSTM-powered music generation pipeline.</p>"
    "<p>LSTM-Powered Symbolic Music Generation</p></div>"
    + bars_html(46, 120, "viz")
    + "</div>"
)



def get_state() -> dict:
    processed = read_json(config.PROCESSED_DIR / config.PREPROCESS_REPORT_FILENAME)
    summary = read_json(config.MODEL_DIR / config.SUMMARY_FILENAME)
    files = model_files_status()
    outputs = sorted(config.OUTPUT_DIR.glob("*.mid"), key=lambda p: p.stat().st_mtime) if config.OUTPUT_DIR.exists() else []
    return {
        "raw_count": len(find_midi_files(config.RAW_DATA_DIR)),
        "processed": processed,
        "summary": summary,
        "files": files,
        "dataset_ready": bool(processed) and files["vocabulary"] and files["processed"],
        "model_ready": files["model_file"] is not None and files["model_config"],
        "outputs": outputs,
    }


def pill(name: str, state: str) -> str:
    icons = {"Ready": "✓", "Not Ready": "○", "Training": "◐", "Error": "⚠"}
    css = state.lower().replace(" ", "")
    return f'<span class="pill {css}">{icons[state]} {esc(name)}: {esc(state)}</span>'


def render_status(slot, state: dict, training: bool = False) -> None:
    error = st.session_state.get("last_error")
    dataset = "Ready" if state["dataset_ready"] else "Not Ready"
    if training:
        model_state = "Training"
    elif error == "training":
        model_state = "Error"
    else:
        model_state = "Ready" if state["model_ready"] else "Not Ready"
    generator = "Error" if error == "generation" else ("Ready" if state["model_ready"] and state["dataset_ready"] else "Not Ready")
    slot.markdown(
        '<div class="status-row">' + pill("Dataset", dataset) + pill("Model", model_state) + pill("Generator", generator) + "</div>",
        unsafe_allow_html=True,
    )


def render_steps(state: dict) -> None:
    done = [state["raw_count"] > 0, state["dataset_ready"], state["model_ready"], bool(state["outputs"]), bool(state["outputs"])]
    names = ["Add MIDI dataset", "Scan / preprocess", "Train model", "Generate music", "Download MIDI"]
    current = next((i for i, flag in enumerate(done) if not flag), None)
    parts = []
    for i, name in enumerate(names):
        css = "done" if done[i] else ("current" if i == current else "")
        mark = "✓" if done[i] else str(i + 1)
        parts.append(f'<div class="step {css}"><b>{mark} · {esc(name)}</b></div>')
    st.markdown('<div class="steps">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def go_page(name: str) -> None:
    st.session_state["page"] = name


def bump(key: str, delta: int | float, low: int | float, high: int | float) -> None:
    """Button callback for the - / + steppers (runs before widgets are drawn)."""
    st.session_state[key] = round(min(max(st.session_state[key] + delta, low), high), 4)


def init_state() -> None:
    d = config.DEFAULT_TRAINING
    st.session_state.setdefault("page", "Dashboard")
    st.session_state.setdefault("tr_seq", d.sequence_length)
    st.session_state.setdefault("tr_batch", d.batch_size)
    st.session_state.setdefault("tr_epochs", d.epochs)


def training_config() -> config.TrainingConfig:
    d = config.DEFAULT_TRAINING
    get = st.session_state.get
    return replace(
        d,
        sequence_length=int(get("tr_seq", d.sequence_length)),
        batch_size=int(get("tr_batch", d.batch_size)),
        epochs=int(get("tr_epochs", d.epochs)),
        validation_split=float(get("tr_val", d.validation_split)),
        lstm_units=int(get("tr_units", d.lstm_units)),
        embedding_dim=int(get("tr_embed", d.embedding_dim)),
        dropout=float(get("tr_dropout", d.dropout)),
        learning_rate=float(get("tr_lr", d.learning_rate)),
    )


def sidebar() -> str:
    d = config.DEFAULT_TRAINING
    current = st.session_state["page"]
    with st.sidebar:
        st.markdown('<div class="brand"><span class="logo">🎵</span> AI Music Generator</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-title">▣ studio</div>', unsafe_allow_html=True)
        for name, icon in NAV:
            st.button(f"{icon}  {name}", key=f"nav_{name.replace(' ', '_')}",
                      type="primary" if name == current else "secondary", on_click=go_page, args=(name,))

        st.divider()
        st.markdown('<div class="side-title">⚙️ Settings</div>', unsafe_allow_html=True)
        for key, label, low, high, step in SETTINGS:
            st.markdown(f'<div class="set-label">{esc(label)}</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f'<div class="set-value">{st.session_state[key]}</div>', unsafe_allow_html=True)
            c2.button("−", key=f"dec_{key}", on_click=bump, args=(key, -step, low, high), help=f"Decrease {label}")
            c3.button("+", key=f"inc_{key}", on_click=bump, args=(key, step, low, high), help=f"Increase {label}")
            st.slider(label, low, high, step=step, key=key, label_visibility="collapsed")

        with st.expander("Advanced Settings"):
            st.number_input("Validation split", 0.05, 0.40, d.validation_split, 0.05, format="%.2f", key="tr_val")
            st.number_input("LSTM units", 16, 512, d.lstm_units, step=16, key="tr_units")
            st.number_input("Embedding size", 8, 256, d.embedding_dim, step=8, key="tr_embed")
            st.number_input("Dropout", 0.0, 0.7, d.dropout, 0.05, format="%.2f", key="tr_dropout")
            st.number_input("Learning rate", 0.0001, 0.01, d.learning_rate, step=0.0005, format="%.4f", key="tr_lr")
        st.caption("Settings apply when you start training. Generation always uses the trained model's own sequence length.")

        st.markdown(
            '<div class="promo">' + bars_html(9, 22, "mini", spread=0.45)
            + "<div><b>Turn your ideas<br>into music</b><span>Powered by AI</span></div></div>",
            unsafe_allow_html=True,
        )
    return current


def page_dashboard(state: dict) -> None:
    render_steps(state)
    processed, summary = state["processed"], state["summary"]
    last = state["outputs"][-1].name if state["outputs"] else "None yet"
    show_cards([
        card("Dataset Files", f"{state['raw_count']} MIDI files", "found in data/raw/"),
        card("Processed Tokens", f"{processed['total_tokens']:,}" if processed else "—", "after preprocessing" if processed else "not preprocessed"),
        card("Vocabulary Size", f"{processed['vocabulary_size']:,} tokens" if processed else "—"),
        card("Model Status", "Ready" if state["model_ready"] else "Not trained"),
        card("Last Generated MIDI", last),
        card("Training Status", f"{summary['epochs_run']} epochs done" if summary else "Not trained yet"),
    ])
    if not state["raw_count"]:
        empty_state("Your music dataset is empty.\nAdd legally obtained MIDI files to data/raw/ to begin.")


def page_dataset(state: dict, slot) -> None:
    with st.container(border=True):
        panel_title("🗄️", "Dataset")
        st.caption(f"Folder: `{config.RAW_DATA_DIR.relative_to(config.PROJECT_ROOT).as_posix()}/`")
        c1, c2 = st.columns(2)
        scan_clicked = c1.button("Scan Dataset", key="btn_scan")
        pre_clicked = c2.button("Preprocess Dataset", key="btn_pre", type="primary", disabled=state["raw_count"] == 0)
        if state["raw_count"] == 0:
            empty_state("No MIDI files found. Add .mid or .midi files to data/raw/.\nYour music dataset is empty. Add legally obtained MIDI files to begin.")
            st.caption("Tip: run `python -m src.sample_data` to create synthetic demo files for a quick test.")
    if scan_clicked:
        try:
            st.session_state["scan"] = scan_dataset().to_dict()
        except Exception as exc:  
            show_error("Dataset scan failed", exc)
    scan = st.session_state.get("scan")
    if scan:
        show_cards([card("MIDI files found", scan["files_found"]), card("Valid files", scan["valid_count"]),
                    card("Skipped / invalid", scan["skipped_count"])])
        if scan["skipped"]:
            st.warning("Some files will be skipped:")
            st.dataframe(scan["skipped"], hide_index=True)
        elif scan["files_found"]:
            st.success("All files passed the quick checks.")
    if pre_clicked:
        bar, text = st.progress(0.0), st.empty()
        try:
            def on_progress(i: int, n: int, name: str) -> None:
                bar.progress(i / n)
                text.text(f"Processing {i}/{n}: {name}")
            with st.spinner("Extracting notes and chords with music21..."):
                report = preprocess_dataset(progress=on_progress)
            st.session_state["last_error"] = None
            record_history("preprocessing", {"files_used": report["files_used"], "tokens": report["total_tokens"],
                                             "vocabulary": report["vocabulary_size"]})
            st.success(f"✔ Preprocessing finished: {report['files_used']} files used, "
                       f"{report['total_tokens']:,} tokens, vocabulary of {report['vocabulary_size']:,}.")
            if report["files_skipped"]:
                st.warning(f"{report['files_skipped']} file(s) were skipped; details are in logs/.")
            render_status(slot, get_state())
        except Exception as exc:
            show_error("Preprocessing failed", exc)
        finally:
            bar.empty()
    elif state["processed"]:
        p = state["processed"]
        st.info(f"Last preprocessing: {p['files_used']} files used, {p['total_tokens']:,} tokens, "
                f"vocabulary {p['vocabulary_size']:,} ({p['created']}).")


def page_training(state: dict, slot) -> None:
    cfg = training_config()
    with st.container(border=True):
        panel_title("⚙️", "Training")
        show_cards([card("Sequence Length", cfg.sequence_length), card("Batch Size", cfg.batch_size),
                    card("Epochs", cfg.epochs), card("Validation Split", cfg.validation_split)])
        if not state["dataset_ready"]:
            empty_state("No processed dataset found.\nComplete the Dataset step (Preprocess Dataset) before training.")
        st.caption("Training time depends on dataset size, CPU/GPU, sequence length, batch size and epochs. "
                   "Training never starts automatically.")
        start = st.button("Start Training", key="btn_train", type="primary", disabled=not state["dataset_ready"])
    if start:
        bar, text = st.progress(0.0), st.empty()
        render_status(slot, state, training=True)
        try:
            def on_progress(epoch: int, total: int, metrics: dict) -> None:
                bar.progress(epoch / total)
                text.text(f"Epoch {epoch}/{total} - loss {metrics.get('loss', 0):.4f} - "
                          f"val loss {metrics.get('val_loss', 0):.4f}")
            with st.spinner("Training the LSTM... this can take a while."):
                summary = train_model(cfg, progress=on_progress)
            st.session_state["last_error"] = None
            record_history("training", {"epochs_run": summary["epochs_run"], "epochs_requested": summary["epochs_requested"],
                                        "best_val_loss": round(summary["best_val_loss"], 4),
                                        "final_train_loss": round(summary["final_train_loss"], 4),
                                        "sequence_length": cfg.sequence_length})
            st.cache_resource.clear()
            st.success("✔ Training completed. Model saved.")
            show_cards([card("Epochs run", summary["epochs_run"]),
                        card("Best validation loss", f"{summary['best_val_loss']:.4f}"),
                        card("Final training loss", f"{summary['final_train_loss']:.4f}")])
        except Exception as exc:
            st.session_state["last_error"] = "training"
            show_error("Training failed", exc)
        finally:
            bar.empty()
            render_status(slot, get_state())
    elif state["summary"]:
        s = state["summary"]
        st.info(f"Last training: {s['epochs_run']} epochs, best validation loss {s['best_val_loss']:.4f}, "
                f"final training loss {s['final_train_loss']:.4f}.")
    plot = config.MODEL_DIR / config.HISTORY_PLOT_FILENAME
    if state["summary"] and plot.exists():
        st.image(str(plot), caption="Training history (from the last run)")


@st.cache_resource(show_spinner="Loading trained model...")
def cached_assets(model_name: str, stamp: float):
    """Load the model once; `stamp` changes whenever the model files change."""
    return load_generation_assets()


@st.cache_data(show_spinner="Rendering audio preview...")
def cached_wav(path_str: str, stamp: float) -> bytes:
    """Render the MIDI to WAV once per file version."""
    return midi_to_wav_bytes(path_str)


def parse_seed(text: str, fixed: bool) -> int | None:
    """Seed rules: a typed number is always used; a ticked box with an empty field uses 42."""
    text = (text or "").strip()
    if text:
        if not text.isdigit() or int(text) > MAX_SEED:
            raise ValidationError(
                "The seed is not valid.",
                reason=f"A seed must be a whole number from 0 to {MAX_SEED}.",
                fix="Type digits only (for example 42), or leave the field empty for a random result.",
            )
        return int(text)
    return 42 if fixed else None


def restore_generation_controls() -> None:
    """Widget values are dropped when you leave this page, so restore them from saved copies."""
    defaults = {"gen_temperature": config.DEFAULT_GENERATION.temperature, "gen_notes": config.DEFAULT_GENERATION.num_notes,
                "gen_fixed": False, "gen_seed": ""}
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get(f"saved_{key}", default)


def save_generation_controls() -> None:
    for key in ("gen_temperature", "gen_notes", "gen_fixed", "gen_seed"):
        st.session_state[f"saved_{key}"] = st.session_state[key]


def share_box(path: Path) -> None:
    box = st.popover("↗ Share") if hasattr(st, "popover") else st.expander("↗ Share")
    with box:
        st.markdown("**Share your composition**")
        st.write("Download the file, then send it by email or chat.")
        st.code(f"outputs/{path.name}", language=None)


def render_output(result: dict) -> None:
    path = Path(result["path"])
    seed_text = result["seed"] if result["seed"] is not None else "random"
    left, right = st.columns([2, 3], gap="large")
    left.markdown(
        '<div class="player-head"><div class="album">🎵</div><div>'
        '<div class="p-title">AI Generated Music</div>'
        f'<div class="p-meta">LSTM • {result["notes"]} notes • Seed: {esc(seed_text)}</div>'
        f'<div class="p-meta">{esc(path.name)}</div></div></div>',
        unsafe_allow_html=True,
    )
    wav = None
    try:
        wav = cached_wav(str(path), path.stat().st_mtime)
        right.audio(wav, format="audio/wav")
    except MusicGenError as exc:
        logger.warning("Audio preview failed: %s", exc.reason)
        right.info("The audio preview is not available, but you can still download the MIDI file.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Download MIDI", data=path.read_bytes(), file_name=path.name, mime="audio/midi", key="dl_midi")
    if wav is not None:
        d2.download_button("⬇ Download WAV", data=wav, file_name=path.with_suffix(".wav").name, mime="audio/wav", key="dl_wav")
    with d3:
        share_box(path)
    st.caption("The preview uses a simple built-in synthesizer, so it sounds basic. "
               "Open the MIDI file in a MIDI player or music software for richer instruments.")


def page_generate(state: dict, slot) -> None:
    restore_generation_controls()
    ready = state["model_ready"] and state["dataset_ready"]
    with st.container(border=True):
        panel_title("🎵", "Generate Music")
        if not ready:
            empty_state("No trained model found.\nComplete preprocessing and training before generating music.")
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            value_row("Temperature", f"{st.session_state['gen_temperature']:.2f}")
            st.slider("Temperature", 0.5, 1.5, step=0.05, key="gen_temperature", label_visibility="collapsed")
            st.markdown('<div class="hint">Lower temperature = more predictable.<br>Higher temperature = more experimental.</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="set-label">Seed</div>', unsafe_allow_html=True)
            st.text_input("Seed", key="gen_seed", placeholder="e.g. 42 (optional)", label_visibility="collapsed")
        with right:
            value_row("Number of Notes", st.session_state["gen_notes"])
            c1, c2, c3 = st.columns([6, 1, 1])
            c1.slider("Number of Notes", 10, 400, step=10, key="gen_notes", label_visibility="collapsed")
            c2.button("−", key="dec_gen_notes", on_click=bump, args=("gen_notes", -10, 10, 400), help="Fewer notes")
            c3.button("+", key="inc_gen_notes", on_click=bump, args=("gen_notes", 10, 10, 400), help="More notes")
            st.checkbox("Use fixed random seed (reproducible)", key="gen_fixed")
            clicked = st.button("▶  Generate Music", key="btn_generate", type="primary", disabled=not ready)

    if clicked:
        try:
            seed = parse_seed(st.session_state["gen_seed"], st.session_state["gen_fixed"])
            model_file = state["files"]["model_file"]
            assets = cached_assets(model_file.name, model_file.stat().st_mtime)
            with st.spinner("Composing..."):
                result = generate_music(num_notes=int(st.session_state["gen_notes"]),
                                        temperature=float(st.session_state["gen_temperature"]),
                                        seed=seed, assets=assets)
            st.session_state["last_result"] = {
                "path": str(result.midi_path), "events": result.num_events, "duration": result.duration_seconds,
                "temperature": result.temperature, "notes": int(st.session_state["gen_notes"]), "seed": result.seed,
            }
            st.session_state["last_error"] = None
            record_history("generation", {"file": result.midi_path.name, "notes": int(st.session_state["gen_notes"]),
                                          "events": result.num_events, "temperature": result.temperature,
                                          "seed": result.seed, "duration_seconds": round(result.duration_seconds)})
            st.success("✔ Generation successful.")
            render_status(slot, get_state())
        except Exception as exc:
            st.session_state["last_error"] = "generation"
            show_error("Generation failed", exc)
            render_status(slot, get_state())

    with st.container(border=True):
        panel_title("🎵", "Generated Output")
        result = st.session_state.get("last_result")
        if result and Path(result["path"]).exists():
            render_output(result)
        else:
            empty_state("Your generated compositions will appear here.")
    save_generation_controls()


def clear_history_action() -> None:
    """Button callback: runs BEFORE the page redraws, so it may safely reset the checkbox."""
    try:
        clear_history()
        st.session_state["hist_confirm"] = False
        st.session_state["hist_notice"] = ("success", "History cleared. Your MIDI files were not deleted.")
    except Exception as exc:
        logger.warning("Could not clear history: %s", exc)
        st.session_state["hist_notice"] = ("error", "Could not clear the history. Check that outputs/ is writable and try again.")


def page_history() -> None:
    panel_title("🕘", "History")
    notice = st.session_state.pop("hist_notice", None)
    if notice:
        (st.success if notice[0] == "success" else st.error)(notice[1])
    entries = load_history()
    outside = untracked_midi_files(entries)
    generations = [e for e in entries if e["kind"] == "generation"]
    trainings = [e for e in entries if e["kind"] == "training"]
    show_cards([
        card("Compositions", len(generations) + len(outside), "MIDI files created"),
        card("Training runs", len(trainings)),
        card("Total actions", len(entries)),
    ])
    if not entries and not outside:
        empty_state("No history yet.\nYour generations, training runs and preprocessing runs will appear here.")
        return

    choice = st.selectbox("Show", ["All", "Compositions", "Training", "Preprocessing"], key="hist_filter")
    wanted = {"All": None, "Compositions": "generation", "Training": "training", "Preprocessing": "preprocessing"}[choice]
    rows = [e for e in entries if wanted is None or e["kind"] == wanted][:50]

    for entry in rows:
        when = esc(entry["time"].replace("T", " "))
        if entry["kind"] == "generation":
            path = config.OUTPUT_DIR / str(entry.get("file", ""))
            seed = entry.get("seed")
            meta = (f"{entry.get('events', '?')} events · temperature {entry.get('temperature', '?')} · "
                    f"seed {seed if seed is not None else 'random'} · about {entry.get('duration_seconds', '?')} s")
            title = f"🎵 {entry.get('file', 'unknown')}"
        elif entry["kind"] == "training":
            path = None
            title = "🧠 Training run"
            meta = (f"{entry.get('epochs_run', '?')}/{entry.get('epochs_requested', '?')} epochs · "
                    f"best validation loss {entry.get('best_val_loss', '?')} · "
                    f"final training loss {entry.get('final_train_loss', '?')}")
        else:
            path = None
            title = "📂 Preprocessing"
            meta = f"{entry.get('files_used', '?')} files · {entry.get('tokens', 0):,} tokens · vocabulary {entry.get('vocabulary', '?')}"
        left, right = st.columns([5, 2])
        left.markdown(f'<div class="track"><div class="t-name">{esc(title)}</div>'
                      f'<div class="t-meta">{when} · {esc(meta)}</div></div>', unsafe_allow_html=True)
        if path is not None:
            if path.exists():
                right.download_button("⬇ Download MIDI", data=path.read_bytes(), file_name=path.name,
                                      mime="audio/midi", key=f"hist_dl_{entry['id']}")
            else:
                right.caption("File no longer in outputs/")

    if wanted in (None, "generation") and outside:
        st.markdown("**Other MIDI files in outputs/ (not in the history list)**")
        for path in outside[:20]:
            left, right = st.columns([5, 2])
            left.markdown(f'<div class="track"><div class="t-name">🎵 {esc(path.name)}</div>'
                          f'<div class="t-meta">{path.stat().st_size / 1024:.1f} KB</div></div>', unsafe_allow_html=True)
            right.download_button("⬇ Download MIDI", data=path.read_bytes(), file_name=path.name,
                                  mime="audio/midi", key=f"hist_out_{path.name}")

    st.divider()
    confirm = st.checkbox("I want to clear the history list (my MIDI files are kept)", key="hist_confirm")
    st.button("Clear History", key="btn_clear_history", disabled=not confirm, on_click=clear_history_action)


def page_about() -> None:
    with st.container(border=True):
        panel_title("ℹ️", "About")
        st.markdown(
            """
**AI Music Generator** is a CodeAlpha Internship project.

**Pipeline:** MIDI files → music21 preprocessing → note/chord tokens → integer encoding →
sliding sequences → LSTM (TensorFlow/Keras) → temperature sampling → MIDI export → audio preview.

* Everything runs locally; no AI API or internet connection is needed.
* Use only legally obtained, public-domain or licensed MIDI data.
* The audio preview is rendered by a simple built-in synthesizer. For richer sound, open the MIDI file in a MIDI player.
"""
        )


def main() -> None:
    init_state()
    page = sidebar()
    state = get_state()
    st.markdown(HERO_HTML, unsafe_allow_html=True)
    slot = st.empty()
    render_status(slot, state)

    if page == "Dashboard":
        page_dashboard(state)
    elif page == "Dataset":
        page_dataset(state, slot)
    elif page == "Training":
        page_training(state, slot)
    elif page == "Generate Music":
        page_generate(state, slot)
    elif page == "History":
        page_history()
    else:
        page_about()


main()