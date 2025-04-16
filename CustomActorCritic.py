from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CustomActorCritic(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        self.total_action_dims = 153  # 152 bets + 1 save
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            **kwargs
        )

        # Final actor head: outputs 153 logits
        self.actor_output = nn.Linear(self.mlp_extractor.latent_dim_pi, self.total_action_dims)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.actor_output(latent_pi)

        proportions = F.softmax(logits, dim=-1)
        save_fraction = proportions[..., -1:]
        bet_fractions = proportions[..., :-1] * (1 - save_fraction)

        dummy_log_std = th.zeros_like(bet_fractions)
        distribution = self.action_dist.proba_distribution(bet_fractions, dummy_log_std)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)

        return actions, values, log_prob


    def _get_action_dist_from_latent(self, latent_pi, latent_vf=None):
        logits = self.actor_output(latent_pi)
        proportions = F.softmax(logits, dim=-1)
        save_fraction = proportions[..., -1:]
        bet_fractions = proportions[..., :-1] * (1 - save_fraction)

        dummy_log_std = th.zeros_like(bet_fractions)
        return self.action_dist.proba_distribution(bet_fractions, dummy_log_std)


    def _predict(self, observation, deterministic=False):
        features = self.extract_features(observation)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.actor_output(latent_pi)
        proportions = F.softmax(logits, dim=-1)
        save_fraction = proportions[..., -1:]
        bet_fractions = proportions[..., :-1] * (1 - save_fraction)
        return bet_fractions
