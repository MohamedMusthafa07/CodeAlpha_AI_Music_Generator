# AI Music Generator - LSTM Music Generation Tool

An AI music generation tool that I built for CodeAlpha Internship - Task 3. It reads a folder of MIDI files, learns note and chord patterns with an LSTM neural network, and generates brand-new MIDI music. The interface is a dark, dashboard-style Streamlit app, and everything runs locally with TensorFlow/Keras, with no paid API, no cloud service and no API keys.

## Project Overview
I wanted to build a music generator that I could train and run on my own laptop, so I used music21 for MIDI parsing and a TensorFlow/Keras LSTM for learning. I spent time on the data pipeline and on handling errors properly. The app validates every MIDI file, skips bad ones instead of crashing, saves the vocabulary to disk, and writes each generated song to a new timestamped file so earlier results are never overwritten.

## Problem Statement
Composing original music is difficult, and many AI music tools need paid services or an internet connection. This project lets a user add MIDI files, train a model and generate new melodies in one place. It also explains problems (no MIDI files, no trained model, too little data) in plain words.

## Features
- Recursive MIDI discovery (`.mid` / `.midi`) with validation of empty, corrupt, oversized and duplicate files
- Bad files are logged and skipped, so they never crash the pipeline
- music21 preprocessing into note and chord tokens (`C4`, `[C4.E4.G4]`)
- Deterministic vocabulary saved to disk and reused for generation
- Sliding-window sequence dataset, memory-capped for student laptops
- Configurable Embedding -> LSTM -> Dropout -> LSTM -> Dropout -> Dense (softmax) model
- Training with ModelCheckpoint, EarlyStopping and ReduceLROnPlateau, plus saved history and plot
- Temperature-controlled sampling with an optional random seed for reproducible output
- Timestamped MIDI output that never overwrites earlier files
- Dark Streamlit dashboard that works offline
- Automated tests with pytest that run in seconds, with no long training

## Technology Stack
| Area | Tools |
|------|-------|
| Language | Python 3.10 - 3.12 |
| UI | Streamlit |
| MIDI processing | music21 |
| Deep learning | TensorFlow / Keras (LSTM) |
| Data and plots | NumPy, Matplotlib |
| Testing | pytest |

## Architecture
```
MIDI files (data/raw)
   -> Validation               (empty, corrupt, oversized, duplicate files)
   -> music21 parsing          (notes and chords become tokens)
   -> Integer encoding         (vocabulary.json)
   -> Sliding sequences        (X, y)
   -> LSTM training            (models/)
   -> Temperature sampling     (next-token prediction)
   -> music21 Stream -> MIDI   (outputs/)
```
The pipeline code lives in the `src/` folder and the interface lives in `app.py`. All settings are kept in one place, `src/config.py`.

## How the Music Generation Works
1. **Preprocessing.** Each MIDI file is parsed with music21. Notes that start at the same moment are merged into one token, so a melody becomes a sequence such as `C4 E4 [C4.E4.G4] D4 ...`.

2. **Encoding and sequences.** Tokens get integer IDs from a sorted vocabulary saved in `data/processed/vocabulary.json`. A sliding window of `sequence_length` tokens is the input and the next token is the target. Windows never cross file boundaries.

3. **Model and generation.** The LSTM predicts the next token. Starting from a real seed window, a token is sampled using the temperature (0.5 = predictable, 1.0 = balanced, 1.5 = experimental), added to the song, and the window slides forward. The tokens are then turned back into music21 notes and written to `outputs/generated_music_YYYYMMDD_HHMMSS.mid`.

## Project Structure
```
CodeAlpha_AI_Music_Generator/
|-- data/raw/            # put your MIDI files here
|-- data/processed/      # encoded tokens + vocabulary (auto-created)
|-- models/              # trained models (auto-created)
|-- outputs/             # generated MIDI files
|-- samples/
|   `-- sample_generated_music.mid   # one sample generated song
|-- logs/                # log files
|-- src/
|   |-- config.py        # all settings in one place
|   |-- preprocessing.py # scan + music21 parsing
|   |-- dataset.py       # vocabulary + sequences
|   |-- model.py         # LSTM model
|   |-- train.py         # training
|   |-- generate.py      # generation + MIDI export
|   |-- midi_utils.py    # MIDI/token helpers, optional playback
|   |-- utils.py         # logging, errors, JSON helpers
|   `-- sample_data.py   # creates synthetic demo MIDI files
|-- tests/               # pytest tests
|-- app.py               # Streamlit interface
|-- requirements.txt
`-- README.md
```

## Dataset Preparation
Put MIDI files (`.mid` / `.midi`, sub-folders allowed) in `data/raw/`.

- Use legally obtained, public-domain or licensed MIDI data only. This repository does not include or download any copyrighted music.
- More files give better results. Start with one style or instrument type (for example piano music by one composer) for the most coherent output.
- To only test the pipeline, create synthetic demo files (no copyright, low musical quality):

```powershell
python -m src.sample_data
```

## Installation
**Windows (PowerShell):**
```powershell
git clone https://github.com/MohamedMusthafa07/CodeAlpha_AI_Music_Generator.git
cd CodeAlpha_AI_Music_Generator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell says "running scripts is disabled on this system", use one of these:
```powershell
# Option A: allow scripts for this terminal window only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Option B: skip activation and call the venv Python directly
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tip: use 64-bit Python 3.10, 3.11 or 3.12. Newer Python versions may not have pre-built TensorFlow packages yet.

## How to Run
Run all commands from the project root folder.

**Step by step (command line):**
```powershell
python -m src.preprocessing
python -m src.train
python -m src.generate
```

Useful options:
```powershell
python -m src.train --epochs 30 --batch-size 128 --sequence-length 50
python -m src.generate --notes 300 --temperature 0.8 --seed 7
python -m src.generate --open
```

**With the Streamlit interface:**
```powershell
streamlit run app.py
```
The app opens at http://localhost:8501. Follow the five steps on the Dashboard: add dataset -> scan/preprocess -> train -> generate -> download MIDI. Training never starts automatically.

## How to Test
```powershell
pytest -q
```
The tests use small fake data, so they need no long training and finish in seconds.


## Error Handling
The tool replies with a short message and a fix instead of a raw error:

| Situation | What the user sees |
|-----------|--------------------|
| No MIDI files found | "No MIDI files were found." Put `.mid` / `.midi` files in `data/raw/` or run `python -m src.sample_data` |
| No trained model | "No trained model was found." Run `python -m src.preprocessing` then `python -m src.train` |
| Too little data | "Only N training sequences can be created." Add more or longer MIDI files, or lower `--sequence-length` |
| Corrupt or duplicate MIDI file | The file is logged and skipped, and the pipeline continues |
| `ModuleNotFoundError: src` | Run commands from the project root using `python -m ...` |
| `ModuleNotFoundError: tensorflow` | Activate `.venv` and run `pip install -r requirements.txt` |
| Out of memory | Lower `--batch-size` or `--sequence-length`, or use fewer MIDI files |
| MIDI file is silent | Your player needs a SoundFont or synth; try VLC or MuseScore |

Detailed technical messages are written to `logs/music_generator.log`.

## Performance Considerations
- Defaults are small enough to train on a normal laptop CPU.
- The sequence dataset is memory-capped, so it does not overload student laptops.
- Early stopping ends training when the validation loss stops improving.
- The vocabulary is saved once and reused, so generation does not need to preprocess again.

## Limitations
- Output is symbolic and uses a fixed note length, so rhythm and dynamics are not modelled.
- Quality depends strongly on the size and consistency of the dataset.
- Chords are detected by notes that start at the same time.
- Very small datasets produce repetitive or random-sounding music.

## What I Learned
- Parsing and cleaning MIDI data with music21
- Turning notes and chords into tokens and building sliding-window sequences
- Building and training an LSTM model with TensorFlow/Keras and its callbacks
- Using temperature sampling to control how creative the generated music is
- Handling bad input files and clear error messages without crashing
- Building a custom-styled Streamlit dashboard
- Testing a machine learning project quickly with pytest
- Publishing and documenting a project on GitHub

## Future Improvements
- Model note durations and velocities
- Multi-instrument generation
- Transformer-based models
- Built-in MIDI-to-audio rendering
- Docker image and CI workflow

## Internship Information
Developed as part of the CodeAlpha Internship - Task 3: AI Music Generation.

## License
MIT - see `LICENSE`.

## Author
**A. Mohamed Musthafa**

- GitHub: [MohamedMusthafa07](https://github.com/MohamedMusthafa07)
- LinkedIn: [mohamedmusthafa07](https://www.linkedin.com/in/mohamedmusthafa07)