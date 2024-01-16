import textworld
import textworld.gym
from textworld.generator import GameOptions, make_game, compile_game

import logging, struct, os, time

def get_random_seed():
    # Generate a seed using os.urandom
    random_seed_bytes = os.urandom(8)  # Get 8 bytes of random data
    seed = struct.unpack("<Q", random_seed_bytes)[0]  # Convert bytes to a single integer
    return seed % 4294967296

def remove_extra_whitespace(text):
    # Split the text into lines, remove trailing whitespace from each line, then rejoin\
    return "\n".join(line.strip() for line in text.splitlines()).strip()

class TextWorldEnv:
    def __init__(self, args):
        self.args = args

        self.reset()

    def reset(self):
        logging.debug("Generating new environment...")

        t0 = time.time()

        self.game = None
        self.game_file = None
        self.env = None

        # Replace seed=0 with random seed
        seed = self.args.seed
        if not seed:
            seed = get_random_seed()

        options = GameOptions()
        options.seeds = seed
        options.nb_rooms = 20
        options.nb_objects = 10
        options.nb_parallel_quests = 5
        options.force_recompile = True
        options.path = f"tw_games/custom_game{seed}.z8"
        options.chaining.min_length = 1
        options.chaining.max_length = 4
        options.chaining.min_breadth = 1
        options.chaining.max_breadth = 4
        options.chaining.min_depth = 1
        options.chaining.max_depth = 4

        try:
            self.game = make_game(options)
            self.game_file = compile_game(self.game, options)
        except Exception as e:
            logging.error(f"Error during game generation: {e}")
            return

        env_id = textworld.gym.register_game(self.game_file, max_episode_steps=self.args.max_episode_steps)
        self.env = textworld.gym.make(env_id)

        obs, _ = self.env.reset()
        obs = remove_extra_whitespace(obs)
        if self.args.verbose:
            self.env.render()

        self.observation = obs
        self.score = 0
        self.done = False
        self.prev_command = ""

        t1 = time.time()

        logging.info(f"Built new environment in {t1 - t0} seconds")

        return obs

    def step(self, command):
        obs, score, done, _ = self.env.step(command)
        obs = remove_extra_whitespace(obs)

        self.prev_command = command
        self.observation = obs
        self.score = score
        self.done = done

        if self.args.verbose:
            self.env.render()

        return obs, score, done
