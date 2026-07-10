# Reinforcement Fine-Tuning (RFT) for ETFs

Implements RL-based fine-tuning (RFT) where the reward is purely task-performance based, not preference aligned. For financial prediction, this optimises directly for downstream portfolio metrics (portfolio return) rather than prediction error. The score combines RFT prediction with momentum.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Pretraining with supervised learning
- RL fine-tuning with PPO
- Reward = portfolio return (task-performance)
- Score = RFT prediction × (1 + last_return)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-reinforcement-fine-tuning-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to RL training)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High positive score → ETF expected to rise tomorrow with RL‑optimised signal.
- Negative score → expected to fall.

## Requirements

See `requirements.txt`.
