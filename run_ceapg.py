import os
import jax
import pickle

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=24'
jax.config.update('jax_platform_name', 'cpu')

#from jax.config import config; config.update("jax_enable_x64", True)

# %load_ext autoreload
# %autoreload 2

from brax import envs
from brax.io import html, model
from brax.training import normalization

import flax
import jax.numpy as jnp
from brax.envs import create_fn

from IPython.display import HTML, clear_output

import optax

import matplotlib.pyplot as plt
import numpy as np

from controllers import GruController, MlpController, LinearController

from ce_apg import do_one_rollout, cem_apg

from functools import partial

def visualize(sys, qps, height=480):
  """Renders a 3D visualization of the environment."""
  return HTML(html.render(sys, qps, height=height))

len(jax.devices())

save_dir = "save_ce_apg_6"

#for env_name in ["inverted_pendulum_swingup", "inverted_double_pendulum_swingup"]: # "acrobot", "inverted_pendulum_swingup", "inverted_double_pendulum_swingup"]:
for env_name in ["acrobot", "inverted_double_pendulum_swingup"]:
    episode_length = 500
    action_repeat = 1
    env_fn = create_fn(env_name = env_name, episode_length=episode_length, action_repeat=action_repeat, batch_size=None, auto_reset=False)
    env = env_fn()

    policy_size = int(2**jnp.ceil(jnp.log2(env.observation_size*4)))
    print(policy_size)
    policy = GruController(env.observation_size, env.action_size, policy_size)
    pickle.dump(policy, open(f"{save_dir}/{env_name}_policy", 'wb'))
    
    for i in range(8):
        inference_fn, params, rewards = cem_apg(env_fn,
                                                200,
                                                key=jax.random.PRNGKey(i),
                                                episode_length = episode_length,
                                                action_repeat = action_repeat,
                                                apg_epochs = 100,
                                                batch_size = 4,
                                                zero_params=True,
                                                truncation_length = None,
                                                learning_rate = 3e-4,
                                                clipping = 1e9,
                                                initial_std = 0.05,
                                                num_elite = 8,
                                                eps = 0.0,
                                                normalize_observations=True,
                                                policy = policy,
                                                learning_schedule = [-3, -6]
                                               )

        model.save_params(f"{save_dir}/{env_name}_params_{i}", params)
        pickle.dump(rewards, open(f"{save_dir}/{env_name}_rewards.pkl{i}", "wb"))

