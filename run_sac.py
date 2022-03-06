from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training import sac
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pickle
from brax.io import model

episode_length = 500
action_repeat = 1
save_dir = "save_sac_6"

#for env_name in ["acrobot"]:
for env_name in ["inverted_pendulum_swingup", "inverted_double_pendulum_swingup", "acrobot"]:  # @param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'walker2d, 'ur5e', 'reacher', bball_1dof]
    env_fn = envs.create_fn(env_name = env_name, action_repeat=action_repeat, batch_size=None, auto_reset=False)

    for i in range(8):
        xdata = []
        ydata = []

        def progress(num_steps, metrics):
          print(f"{env_name} {i} {num_steps}: {metrics['eval/episode_reward']}")
          ydata.append(metrics['eval/episode_reward'])
          xdata.append(num_steps)
          # clear_output(wait=True)
          # plt.xlabel('# environment steps')
          # plt.ylabel('reward per episode')
          # plt.plot(xdata, ydata)
          # plt.show()
        
        
        inference_fn, params, metrics = sac.train(
            environment_fn=envs.create_fn(env_name, auto_reset=False, episode_length=episode_length),
            num_timesteps = 2_000_000,
            episode_length = episode_length, normalize_observations = True,
            action_repeat = action_repeat,
            discounting = 0.95, learning_rate = 1e-3,
            num_envs = 64, batch_size = 128, seed = i,
            progress_fn = progress
        )

        model.save_params(f"{save_dir}/{env_name}_params_{i}", params)
        pickle.dump(metrics, open(f"{save_dir}/{env_name}_metrics{i}.pkl", 'wb'))
        pickle.dump(ydata, open(f"{save_dir}/{env_name}_rewards.pkl{i}", 'wb'))


  
