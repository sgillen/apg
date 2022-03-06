from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import apg
import os
import jax
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pickle
from brax.io import model

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=24'
jax.config.update('jax_platform_name', 'cpu')


episode_length=500
action_repeat=1
save_dir = 'save_bapg_6'



for env_name in ["inverted_pendulum_swingup", "inverted_double_pendulum_swingup", "acrobot"]:  # @param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'walker2d, 'ur5e', 'reacher', bball_1dof]

    env_fn = envs.create_fn(env_name = env_name, action_repeat=action_repeat, batch_size=None, auto_reset=False)

    xdata = []
    ydata = []

    for i in range(8):
        xdata = []
        ydata = []

        def progress(num_steps, metrics):
            print(f"{env_name} {i} {num_steps}: {metrics['eval/episode_reward']}")
            xdata.append(num_steps)
            ydata.append(metrics['eval/episode_reward'])
            # clear_output(wait=True)
            # plt.xlabel('# environment steps')
            # plt.ylabel('reward per episode')
            # plt.plot(xdata, ydata)
            # plt.show()

        inference_fn, params, metrics = apg.train(
            environment_fn=env_fn,
            episode_length=episode_length,
            action_repeat=action_repeat,
            num_envs=24,
            num_eval_envs=1,
            learning_rate=5e-4,
            normalize_observations=True,
            log_frequency=200*100//24,
            truncation_length=None,
            progress_fn=progress
        )
    
        model.save_params(f"{save_dir}/{env_name}_params_{i}", params)
        pickle.dump(metrics, open(f"{save_dir}/{env_name}_metrics{i}.pkl", 'wb'))
        pickle.dump(ydata, open(f"{save_dir}/{env_name}_rewards.pkl{i}", 'wb'))
