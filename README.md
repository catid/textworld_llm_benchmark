# TextWorld LLM Benchmark

Many people (myself included) consider intelligence to be equivalent to the ability to play games well.  Games require reasoning, strategy, and lots of other human factors that are not usually tested on LLM benchmarks that can be solved by "training on the test set".

Also consider that right now there is no general-purpose AI that is super-human at playing games.  There are only specialized AIs that are hand-coded for specific games that exceed human ability.  But given an unfamiliar game, humans are still superior at learning the rules and succeeding.

Introducing: A hard AI reasoning benchmark that should be difficult or impossible to cheat at, because it's generated randomly each time!

It uses a *lot* of LLM inference so to avoid a $100 bill from OpenAI, I suggest self-hosting a model.
The code does use OpenAI API though if you want to see what GPT-4 or other models would do on this test.
Right now, the easiest way to host Mixtral on a Linux server with Nvidia GPUs is this method: https://github.com/catid/oaimixtral


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
python llm_play_textworld.py --openai_api_key "14d78630027e15de243c8b3b489a91fa" --openai_base_url "http://devnuc.lan:5000/v1"
```

You can specify the experiment parameters like this:

```bash
python llm_play_textworld.py --num_tests 100 --parallel 16 --max_episode_steps 50 --openai_api_key "14d78630027e15de243c8b3b489a91fa" --openai_base_url "http://devnuc.lan:5000/v1"
```

The defaults are 100 tests at up to 50 steps, and 16 in parallel at a time, but you may want to run more or fewer in parallel based on your setup.

You can also adjust the model temperature and max output tokens.

With 6 Mixtral server setup, this finishes in about 30 minutes.


## Results

Mixtral 8x7B model results:

```
```

I think this is a pretty good benchmark because the AI is able to score some points, but fails to complete the game most of the time.  There's plenty of room for new AIs to improve on this benchmark.
