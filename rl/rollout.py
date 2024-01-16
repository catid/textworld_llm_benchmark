# Recording of trajectories taken by the policy

################################################################################
# Trajectory

class Trajectory:
    def __init__(self):
        # List of observations preceeding the actions, stored as strings
        self.observations_strings = []
        # Observation embeddings, cached to avoid recomputing these
        self.observations_embeddings = []

        # Action taken in response to observation index i
        self.actions_string = []
        # Cached embedding of actions
        self.actions_embedding = []

        # Reward at index i is the reward for action taken at index i
        self.rewards = []

        self.Done = False
        self.Truncated = False

    # Record next event in the trajectory
    def RecordNextEvent(self,
                     observations_strings, observations_embeddings,
                     action_string, action_embedding, reward):
        self.observations_strings.append(observations_strings)
        self.observations_embeddings.append(observations_embeddings)
        self.actions_string.append(action_string)
        self.actions_embedding.append(action_embedding)
        self.rewards.append(reward)

    def RecordEnd(self, done, truncated):
        self.Done = done
        self.Truncated = truncated


################################################################################
# Batch

class Batch:
