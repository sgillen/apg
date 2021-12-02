import jax
import jax.numpy as jnp
import flax
import optax
from functools import partial
from brax.training import normalization

## Common



### apg


@jax.jit
def add_noise(params, noise_std, key):
    num_vars = len(jax.tree_leaves(params))
    treedef = jax.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), params,
        jax.tree_unflatten(treedef, all_keys))
    
    params_with_noise = jax.tree_multimap(lambda g, n: g + n * noise_std,
                                      params, noise)
    # anti_params_with_noise = jax.tree_multimap(
    #     lambda g, n: g - n * perturbation_std, params, noise)
    
    return params_with_noise, noise


@partial(jax.jit, static_argnums=(0,1,2))
def do_local_apg(num_epochs, env_fn, policy_apply, normalizer_params, policy_params, key, learning_rate=1e-3, episode_length=1000, action_repeat=1, normalize_observations=True):

    env = env_fn()
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(policy_params)

    _, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(
          env.observation_size, normalize_observations, num_leading_batch_dims=1)



    
    def do_rnn_rollout(policy_params, normalizer_params, key):
        init_state = env.reset(key)
        h0 = jnp.zeros_like(init_state.obs)

        def do_one_rnn_step(carry, step_idx):
            state, h, policy_params, normalizer_params  = carry

            normed_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
            h1 , actions = policy_apply(policy_params, h, normed_obs)
            nstate = env.step(state, actions)    
            #h1 = jax.lax.cond(nstate.done, lambda x: jnp.zeros_like(h1), lambda x: h1, None)
            return (jax.lax.stop_gradient(nstate), h1, policy_params, normalizer_params), (nstate.reward,state.obs, actions, nstate)

        
        _, (rewards, obs, acts, states) = jax.lax.scan(
            do_one_rnn_step, (init_state, h0, policy_params, normalizer_params),
            (jnp.array(range(episode_length // action_repeat))),
            length=episode_length // action_repeat)
        
        return rewards, obs, acts, states

    
    def sum_rnn_loss(policy_params, normalizer_params, key):
        (rewards, obs, acts,nstate) = do_rnn_rollout(policy_params, normalizer_params, key)
    
        return -jnp.sum(rewards), (rewards, obs)
    


    grad_fn = jax.jit(jax.grad(sum_rnn_loss, has_aux=True))
    
    def do_one_apg_iter(carry, epoch):
        optimizer_state, key, policy_params, normalizer_params = carry
        key, reset_key = jax.random.split(key)
        init_state = env.reset(reset_key)

        grads, (rewards, obs) = grad_fn(policy_params, normalizer_params, reset_key)
        param_updates, optimizer_state = optimizer.update(grads, optimizer_state)
        policy_params = optax.apply_updates(policy_params, param_updates)

        normalizer_params = obs_normalizer_update_fn(normalizer_params, obs)

        return (optimizer_state, key, policy_params, normalizer_params), jnp.sum(rewards)



    
    (optimizer_state, key, policy_params, normalizer_params), rewards = jax.lax.scan(do_one_apg_iter, (optimizer_state, key, policy_params, normalizer_params), (jnp.array(range(num_epochs))))

    return policy_params, normalizer_params, rewards


## basin hopping
