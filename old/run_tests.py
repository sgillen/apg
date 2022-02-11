import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=12'

from brax import envs
from brax.io import html
from brax.training import normalization

import flax
import jax
import jax.numpy as jnp
from brax.envs import create_fn

from IPython.display import HTML, clear_output

import optax

import matplotlib.pyplot as plt
import numpy as np

from controllers import GruController, MlpController
from common import do_local_apg, add_guassian_noise, add_uniform_noise, add_uniform_and_pareto_noise, add_sym_pareto_noise, do_one_rollout

from functools import partial

jax.config.update('jax_platform_name', 'cpu')

def visualize(sys, qps, height=480):
  """Renders a 3D visualization of the environment."""
  return HTML(html.render(sys, qps, height=height))


episode_length = 500
action_repeat = 1
batch_size = jax.local_device_count()
#noise_std = 0.2

truncation_length = 10
noise_beta = 2.0

apg_epochs = 100
batch_size = 4
learning_rate = 5e-4
clipping = 1e9

normalize_observations=True

env_name = "reacher"  # @param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'walker2d, 'ur5e', 'reacher', bball_1dof]
env_fn = create_fn(env_name = env_name, action_repeat=action_repeat, batch_size=None, auto_reset=False)
env = env_fn()

key = jax.random.PRNGKey(0)
reset_keys = jax.random.split(key, num=jax.local_device_count())
_, model_key = jax.random.split(reset_keys[0])
noise_keys = jax.random.split(model_key, num=jax.local_device_count())

policy = GruController(env.observation_size,env.action_size,128)

add_noise_pmap = jax.pmap(add_uniform_noise, in_axes=(None,None,0))
add_pareto_noise_pmap = jax.pmap(add_uniform_and_pareto_noise, in_axes=(None,None,None,0))

do_apg_pmap = jax.pmap(do_local_apg, in_axes = (None,None,None,None,0,0,None,None,None,None,None,None), static_broadcasted_argnums=(0,1,2,6,7,8,9,10,11,12))
do_rollout_pmap = jax.pmap(do_one_rollout, in_axes = (None,None,None,0,0,None,None,None), static_broadcasted_argnums=(0,1,5,6,7))

for noise_scale in [0.0, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0]:
       normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(env.observation_size, normalize_observations, num_leading_batch_dims=1)
       print(f"{noise_scale}")
       init_states = jax.pmap(env.reset)(reset_keys)
       x0 = init_states.obs
       h0 = jnp.zeros(env.observation_size)

       policy_params = policy.init(model_key, h0, x0)

       best_reward = -float('inf')
       meta_rewards_list = []

       for i in range(40):
          noise_keys = jax.random.split(noise_keys[0], num=jax.local_device_count())
          train_keys = jax.random.split(noise_keys[0], num=jax.local_device_count())

          #policy_params_with_noise, noise = add_noise_pmap(policy_params, noise_std, noise_keys)
          policy_params_with_noise, noise1,noise2 = add_pareto_noise_pmap(policy_params, noise_beta, noise_scale, noise_keys)

          rewards_before, obs, acts, states_before = do_rollout_pmap(env_fn, policy.apply, normalizer_params, policy_params_with_noise, train_keys, episode_length, action_repeat, normalize_observations)
          policy_params_trained, rewards_lists = do_apg_pmap(apg_epochs, env_fn, policy.apply, normalizer_params, policy_params_with_noise, train_keys, learning_rate, episode_length, action_repeat, normalize_observations, batch_size, clipping, truncation_length)
          rewards_after, obs, acts, states_after = do_rollout_pmap(env_fn, policy.apply, normalizer_params, policy_params_trained, train_keys, episode_length, action_repeat, normalize_observations)

          #print(jnp.any(policy_params_trained['params']['Dense_1']['kernel'] - policy_params_with_noise['params']['Dense_1']['kernel']))

          top_idx = sorted(range(len(rewards_lists)), key=lambda k: jnp.mean(rewards_lists[k][-5:]), reverse=True)

          _, params_def = jax.tree_flatten(policy_params)
          params_flat, _ = jax.tree_flatten(policy_params_trained)
          top_params_flat = [param[top_idx[0]] for param in params_flat]
          top_params = jax.tree_unflatten(params_def, top_params_flat)

          #     _, norm_def = jax.tree_flatten(normalizer_params)
          #     norm_flat, _ = jax.tree_flatten(normalizer_params_all)
          #     top_norm_flat = [param[top_idx[0]] for param in norm_flat]
          #     top_norms = jax.tree_unflatten(norm_def, top_norm_flat)

          noise_beta -= .1

          if jnp.mean(rewards_lists[top_idx[0]][-5:]) > best_reward:
              noise_beta = 2.0
              policy_params = top_params
              top_normalizer_params = normalizer_params
              best_reward = jnp.mean(rewards_lists[top_idx[0]][-5:])


          meta_rewards_list.append(best_reward)
          normalizer_params = obs_normalizer_update_fn(normalizer_params, obs[top_idx[0],:])

          print(f" Iteration {i} --------------------------------\n")

          for j in range(len(top_idx)):
              done_idx = jnp.where(states_before.done[top_idx[j], :], size=1)[0].item()
              if done_idx == 0:
                  done_idx = rewards_before.shape[-1]
              rewards_sum_before = jnp.sum(rewards_before[top_idx[j],:done_idx])

              done_idx = jnp.where(states_after.done[top_idx[j], :], size=1)[0].item()
              if done_idx == 0:
                  done_idx = rewards_after.shape[-1]
              rewards_sum_after = jnp.sum(rewards_after[top_idx[j],:done_idx])

              print(f"{j} : reward: {rewards_sum_before} -> {jnp.mean(rewards_lists[top_idx[j]][-5:])}  |  {rewards_sum_after}\n")

          #print(f"{i} : best reward: {rewards_sum_before} -> {rewards_lists[top_idx[0]][-1]}  |  {rewards_sum_after}")

          print(f"Best reward so far: {best_reward}\n")
          print('--------------------------------------\n')

          if jnp.any(jnp.isnan(rewards_lists)):
            break

print("\n\n=====================================\n\n")
