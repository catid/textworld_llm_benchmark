# textworld_solver

This is a hard AI reasoning benchmark that should be difficult or impossible to cheat at.

It uses a *lot* of LLM inference so to avoid a $100 bill from OpenAI, I suggest self-hosting a model: https://github.com/catid/oaimixtral

The code does use OpenAI API though if you want to see what GPT-4 or other models would do on this test.

## Setup

Install Conda: https://docs.conda.io/projects/miniconda/en/latest/

```bash
conda create -n tws python=3.10
conda activate tws

git clone https://github.com/catid/textworld_solver.git
cd textworld_solver

pip install -r requirements.txt
```

## Usage

```bash
conda activate tws

# Launch the experiment!
python llm_play_textworld.py
```

You can specify the experiment parameters like this:

```bash
python llm_play_textworld.py --num_tests 100 --parallel 16 --max_episode_steps 50
```

The defaults are 100 tests at up to 50 steps, and 16 in parallel at a time, but you may want to run more or fewer in parallel based on your setup.

With 6 Mixtral server setup, this finishes in about 30 minutes.
