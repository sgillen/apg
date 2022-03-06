from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import ppo
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pickle
from brax.io import model

episode_length = 500
action_repeat = 1
save_dir = "save_ppo_6"

for env_name in ["inverted_pendulum_swingup", "inverted_double_pendulum_swingup", "acrobot"]:
    env_fn = envs.create_fn(env_name = env_name, action_repeat=action_repeat, batch_size=None, auto_reset=False)

    for i in range(8):
        xdata = []
        ydata = []

        def progress(num_steps, metrics):
            xdata.append(num_steps)
            ydata.append(metrics['eval/episode_reward'])
            # clear_output(wait=True)
            # plt.xlabel('# environment steps')
            # plt.ylabel('reward per episode')
            # plt.plot(xdata, ydata)
            # plt.show()

        inference_fn, params, metrics = ppo.train(
            environment_fn=envs.create_fn(env_name, auto_reset=True),
            num_timesteps = 80_000_000, log_frequency = 20,
            reward_scaling = 1, episode_length = episode_length, normalize_observations = True,
            action_repeat = action_repeat, unroll_length = 50, num_minibatches = 32,
            num_update_epochs = 8, discounting = 0.99, learning_rate = 3e-4,
            entropy_cost = 1e-3, num_envs = 512, batch_size = 256, seed = i,
            progress_fn = progress
        )
        model.save_params(f"{save_dir}/{env_name}_params_{i}", params)
        pickle.dump(metrics, open(f"{save_dir}/{env_name}_metrics{i}.pkl", 'wb'))
        pickle.dump(ydata, open(f"{save_dir}/{env_name}_rewards.pkl{i}", 'wb'))
