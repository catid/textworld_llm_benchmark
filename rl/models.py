# Model-free Reinforcement Learning

import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################################
# Embeddings

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# This will return (len(sentences), embedding_dim) shaped tensors
def get_sentences_embedding(sentences):
    embeddings = embedding_model.encode(sentences)
    return embeddings

def get_embedding_dim():
    embeddings = get_sentences_embedding(["What's your embedding dimension?"])
    return embeddings.shape[1]


################################################################################
# Actor Model

from lru import LRU

# Action scoring model scores a batch of potential actions given a list of
# observations, assigning a number -1..1 to each action.
class ActionScoringModel(nn.Module):
    def __init__(self, d_embed=384, d_obs_hidden=64, d_obs_out=64, d_mlp=64):
        super().__init__()

        # Encodes a variable-length set of observation sentences (as embeddings)
        # into a learned latent space.
        self.observation_head = LRU(d_in=d_embed, d_hidden=d_obs_hidden, d_out=d_obs_out)

        # Learned scoring 
        self.action_score_network = nn.Sequential(
            nn.Linear(d_embed + d_obs_out, d_mlp),
            nn.Tanh(),
            nn.Linear(d_mlp, d_mlp//2),
            nn.Tanh(),
            nn.Linear(d_mlp//2, 1),
            nn.Tanh(),
        )

    # observations_embeddings: tensor of shape (batch, len(observations), d_embed)
    # actions_embeddings: tensor of shape (batch, len(actions), d_embed)
    def forward(self, observations_embeddings, actions_embeddings):
        # Latents for observations (batch, d_obs_out)
        latent_obs = self.observation_head(observations_embeddings)

        # Max-pool the latents (batch, d_obs_out)
        # The idea here is to simply keep the strongest latent signals from each
        # observation, with the intuition that each signal may correspond to a
        # learned condition
        latent_obs, _ = torch.max(latent_obs, dim=1)

        # Expand the latent obs so we duplicate it so there's one per action
        # Combine latent_obs and actions_embeddings
        inputs = torch.cat((latent_obs.unsqueeze(1).expand(-1, actions_embeddings.size(1), -1), actions_embeddings), dim=-1)

        # Produce one score per action per batch (batch, len(actions), 1)
        return self.action_score_network(inputs)


################################################################################
# Critic Model

from lru import LRU

# Critic model is modeling expectation of reward to check if we're improving our
# expectation of reward during online training.
class ObservationExpectationModel(nn.Module):
    def __init__(self, d_embed=384, d_obs_hidden=64, d_obs_out=48, d_mlp=32):
        super().__init__()

        # Encodes a variable-length set of observation sentences (as embeddings)
        # into a learned latent space.
        self.observation_head = LRU(d_in=d_embed, d_hidden=d_obs_hidden, d_out=d_obs_out)

        # Learned scoring 
        self.expected_score_network = nn.Sequential(
            nn.Linear(d_embed + d_obs_out, d_mlp),
            nn.Tanh(),
            nn.Linear(d_mlp, d_mlp//2),
            nn.Tanh(),
            nn.Linear(d_mlp//2, 1),
            nn.Tanh(),
        )

    # observations_embeddings: tensor of shape (batch, len(observations), d_embed)
    # actions_embeddings: tensor of shape (batch, len(actions), d_embed)
    def forward(self, observations_embeddings, actions_embeddings):
        # Latents for observations (batch, len(obs), d_obs_out)
        latent_obs = self.observation_head(observations_embeddings)

        # Max-pool the latents (batch, d_obs_out)
        # The idea here is to simply keep the strongest latent signals from each
        # observation, with the intuition that each signal may correspond to a
        # learned condition
        latent_obs, _ = torch.max(latent_obs, dim=1)

        # Expand the latent obs so we duplicate it so there's one per action
        # Combine latent_obs and actions_embeddings
        inputs = torch.cat((latent_obs.unsqueeze(1).expand(-1, actions_embeddings.size(1), -1), actions_embeddings), dim=-1)

        # Produce one score per action per batch (batch, len(actions), 1)
        return self.expected_score_network(inputs)
