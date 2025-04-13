# 🤖 OpenAI Agents + Chainlit Chatbot

A developer playground to explore and experiment with **OpenAI Agents SDK** using **Chainlit** as the frontend framework. We've included essential setup with [`uv`](https://github.com/astral-sh/uv) for blazing-fast package management and reproducibility.

> ✅ This repo also includes **advanced Python concepts** to help you better understand how OpenAI Agents work under the hood.

---

## 🚀 Quickstart

### 1. Clone the Repo

```bash
git clone https://github.com/AmmarAamir786/openai_agents_sdk
cd openai_agents_sdk
```

---

### 2. Install Dependencies with `uv`

Run the following commands in your terminal one by one:

```bash
uv venv
source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows

uv add openai-agents
uv add dotenv
uv add chainlit
```

---

### 3. Set Environment Variables

Create a `.env` file in the root directory and set your API keys and required configs:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 💬 Running the Chatbot

To launch the Chainlit chatbot interface, use:

```bash
uv run chainlit run chatbot.py -w
```

---

## 🧪 Running Specific Agent Files

We are using structured demos under the `_01/` directory. You can run them individually like so:

### 🔁 Agent File Starting Point

```bash
uv run _01/01_sync.py
```

### 💡 Chainlit-Integrated Agent File

```bash
uv run chainlit run _01/03_chainlit.py -w
```

> 🔄 **Note:** Change the filename after `_01/` to the specific chainlit file you want to run.

---

## ⚠️ Troubleshooting

If the code doesn't work out-of-the-box, try aligning your environment by using the **exact `uv` and Python versions** I used. Some libraries and features (especially experimental ones from OpenAI) are version-sensitive.

📌 Check your `pyproject.toml` or the generated lock files (`.uv/`) for pinned package versions.