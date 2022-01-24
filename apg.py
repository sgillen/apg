import jax
from jax import lax
import jax.numpy as jnp
import flax

import optax
from functools import partial
from brax.training import normalization
from IPython.display import HTML, clear_output
import matplotlib.pyplot as plt


from controllers import GruController, MlpController, LinearController

@jax.jit
def add_guassian_noise_mixed_std(params, noise_std, key):
    num_vars = len(jax.tree_leaves(params))
    treedef = jax.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_multimap(
        lambda g, k, s: s*jax.random.normal(k, shape=g.shape, dtype=g.dtype),
        params,
        jax.tree_unflatten(treedef, all_keys),
        noise_std
    )
    
    params_with_noise = jax.tree_multimap(lambda g, n: g + n,
                                      params, noise)
    # anti_params_with_noise = jax.tree_multimap(
    #     lambda g, n: g - n * perturbation_std, params, noise)
    
    return params_with_noise, noise


#@partial(jax.jit, static_argnums=(0,1,2,6,7,8,9,10,11,12))
def do_local_apg(num_epochs, env_fn, policy_apply, normalizer_params, policy_params, key, learning_rate=1e-3, episode_length=1000, action_repeat=1, normalize_observations=True, batch_size=1, clipping=1e9, truncation_length=None):

    env = env_fn()
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(policy_params)

    _, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(
          env.observation_size, normalize_observations, num_leading_batch_dims=1)

    clip_init, clip_update = optax.adaptive_grad_clip(clipping, eps=0.001)
    clip_state = clip_init(policy_params)

    @jax.jit
    def do_training_rollout(policy_params, key):
        init_state = env.reset(key)
        h0 = jnp.zeros_like(init_state.obs)

        def do_one_rnn_step(carry):
            state, h, reward_sum, step_index  = carry

            normed_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
            h1 , actions = policy_apply(policy_params, h, normed_obs)
            nstate = env.step(state, actions)

            #nstate = jax.lax.cond(True, lambda s,a: env.step(s, a), lambda s,a: s, state,actions)
            #h1 = jax.lax.cond(nstate.done, lambda x: jnp.zeros_like(h1), lambda x: h1, None)
            #jax.lax.stop_gradient(nstate)

            reward_sum += nstate.reward
            
            if truncation_length is not None and truncation_length > 0:
                nstate = jax.lax.cond(
                    jnp.mod(step_index + 1, truncation_length) == 0.,
                    jax.lax.stop_gradient, lambda x: x, nstate)

            step_index += 1
            return (nstate, h1, reward_sum, step_index), None
            #return (nstate, h1, policy_params, normalizer_params, reward_sum), None

        def noop(carry):
            return carry, None

        def body_fn(carry, x):
            done = jnp.any(carry[0].done)
            return jax.lax.cond(done, noop, do_one_rnn_step, carry)
        
        carry , _  = jax.lax.scan(
            body_fn, (init_state, h0, 0.0, 0),
            None,
            length=episode_length // action_repeat)

        reward_sum = carry[-2]
        
        return reward_sum    
    
    vmap_rnn_rollout = jax.vmap(do_training_rollout, in_axes=(None, 0))

    @jax.jit
    def sum_rnn_loss(policy_params, normalizer_params, key):
        keys = jax.random.split(key, num=batch_size)
        rewards_sums = vmap_rnn_rollout(policy_params, keys)
        return -jnp.mean(rewards_sums)

    grad_fn = jax.value_and_grad(sum_rnn_loss)

    @jax.jit
    def do_one_apg_iter(carry, epoch):
        optimizer_state, clip_state, key, policy_params = carry

        #key, reset_key = jax.random.split(key) # using the same key for each APG run works well, even when evaluating on many keys afterwards
        reset_key = key
        init_state = env.reset(reset_key)

        rewards, grads = grad_fn(policy_params, normalizer_params, reset_key)
        param_updates, optimizer_state = optimizer.update(grads, optimizer_state)
        param_updates, clip_state = clip_update(param_updates, clip_state, policy_params)
        policy_params = optax.apply_updates(policy_params, param_updates)

        #normalizer_params = obs_normalizer_update_fn(normalizer_params, obs)

        return (optimizer_state, clip_state, key, policy_params), -rewards
    
    (optimizer_state, clip_state, key, policy_params2), rewards = jax.lax.scan(do_one_apg_iter, (optimizer_state, clip_state, key, policy_params), (jnp.array(range(num_epochs))))

    return policy_params2, rewards


def make_inference_fn(observation_size, action_size, normalize_observations, policy):

    _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
    observation_size, normalize_observations)

    def inference_fn(params, h0, obs):
        normalizer_params, policy_params = params
        obs = obs_normalizer_apply_fn(normalizer_params, obs)
        h1, action = policy.apply(policy_params, h0, obs)
        return h1, action

    return inference_fn


def apg(env_fn,
            apg_epochs,
            episode_length = 500,
            action_repeat = 1,
            batch_size = 1,
            key = jax.random.PRNGKey(0),
            normalize_observations=True,
            truncation_length = None,
            learning_rate = 5e-4,
            clipping = 1e9,
            eps = 0.0,
            policy = None,
            print_freq = 1,
            ):

    env = env_fn()

    if policy is None:
        # Select policy size that is the closest power of 2 larger than 4x the observation size
        policy_size = int(2**jnp.ceil(jnp.log2(env.observation_size*4)))
        policy = GruController(env.observation_size, env.action_size, policy_size)

    num_directions = jax.local_device_count()
    reset_keys = jax.random.split(key, num=num_directions)
    noise_keys = jax.random.split(reset_keys[0], num=num_directions)
    _, model_key = jax.random.split(noise_keys[0])

    normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(
        env.observation_size, normalize_observations, num_leading_batch_dims=1)

    add_noise_pmap = jax.pmap(add_guassian_noise_mixed_std, in_axes=(None,None,0))
    do_apg_pmap = jax.pmap(do_local_apg, in_axes = (None,None,None,None,0,0,None,None,None,None,None,None), static_broadcasted_argnums=(0,1,2,6,7,8,9,10,11,12))
    do_rollout_pmap = jax.pmap(do_one_rollout, in_axes = (None,None,None,0,0,None,None,None), static_broadcasted_argnums=(0,1,5,6,7))
    
    init_states = jax.pmap(env.reset)(reset_keys)
    x0 = init_states.obs
    h0 = jnp.zeros(env.observation_size)
    
    policy_params = policy.init(model_key, h0, x0)

    policy_params_flat, policy_params_def = jax.tree_flatten(policy_params)
    noise_std = jax.tree_unflatten(policy_params_def, [jnp.ones_like(p)*initial_std for p in policy_params_flat])
    best_reward_list = []

    # tau = (total_epochs-i)/total_epochs
    # eps = tau*eps + (1 - tau)*eps_end

    noise_keys = jax.random.split(noise_keys[0], num=num_directions)
    train_keys = jax.random.split(noise_keys[0], num=num_directions)

    rewards_before_noise , obs, _, states_after_cem = do_one_rollout(env_fn, policy.apply, normalizer_params, policy_params, train_keys[0], episode_length, action_repeat, normalize_observations)
    normalizer_params = obs_normalizer_update_fn(normalizer_params, obs)

    policy_params_with_noise, noise = add_noise_pmap(policy_params, noise_std, noise_keys)
    rewards_before_apg, obs, _, states_before_apg = do_rollout_pmap(env_fn, policy.apply, normalizer_params, policy_params_with_noise, train_keys, episode_length, action_repeat, normalize_observations)

    policy_params_with_noise, rewards_lists = do_apg_pmap(apg_epochs, env_fn, policy.apply, normalizer_params, policy_params_with_noise, train_keys, learning_rate, episode_length, action_repeat, normalize_observations, batch_size, clipping, truncation_length)

    reward_sums = [jnp.mean(rew[-5:]) for rew in rewards_lists]
    top_idx = sorted(range(len(reward_sums)), key=lambda k: reward_sums[k], reverse=True)

    best_reward_list.append(reward_sums[top_idx[0]])    

    inference_fn = make_inference_fn(env.observation_size, env.action_size, normalize_observations, policy)
    params = normalizer_params, policy_params
    return inference_fn, params, best_reward_list



@partial(jax.jit, static_argnums=(0,1,5,6,7))
def do_one_rollout(env_fn, policy_apply, normalizer_params, policy_params, key, episode_length=1000, action_repeat=1, normalize_observations=True):

    env = env_fn()
    _, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(
          env.observation_size, normalize_observations, num_leading_batch_dims=1)

    init_state = env.reset(key)
    h0 = jnp.zeros_like(init_state.obs)

    @jax.jit
    def do_one_rnn_step(carry, step_idx):
        state, h  = carry
        
        normed_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
        h1 , actions = policy_apply(policy_params, h, normed_obs)
        nstate = env.step(state, actions)    
        #h1 = jax.lax.cond(nstate.done, lambda x: jnp.zeros_like(h1), lambda x: h1, None)
        #jax.lax.stop_gradient(nstate)
        return (jax.lax.stop_gradient(nstate), h1), (nstate.reward,state.obs, actions, jax.lax.stop_gradient(nstate))
        #return (nstate, h1, policy_params, normalizer_params), (nstate.reward,state.obs, actions, nstate)


    _, (rewards, obs, acts, states) = jax.lax.scan(
        do_one_rnn_step, (init_state, h0),
        (jnp.array(range(episode_length // action_repeat))),
        length=episode_length // action_repeat)
        
    return rewards, obs, acts, states
