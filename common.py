import jax
from jax import lax
import jax.numpy as jnp
import flax

import optax
from functools import partial
from brax.training import normalization

## Common



### apg



@jax.jit
def add_sym_pareto_noise(params, a, scale, key):
    num_vars = len(jax.tree_leaves(params))
    treedef = jax.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)

    # def sym_pareto(g, k):
    #     s = 2*(jax.random.bernoulli(k, shape = g.shape) - .5) 
    #     r = jax.random.uniform(k, shape = g.shape, dtype=g.dtype)
    #     return s*(a - 1.0)/lax.pow(r, a)

    def sym_pareto(g,k):
        s = 2*(jax.random.bernoulli(k, shape = g.shape) - .5) 
        return scale*s*(jax.random.pareto(k, a, shape=g.shape, dtype=g.dtype) - 1.0)
    
    noise = jax.tree_multimap(
        lambda g, k: sym_pareto(g, k), params,
        jax.tree_unflatten(treedef, all_keys))
    
    params_with_noise = jax.tree_multimap(lambda g, n: g + n*scale,
                                      params, noise)
    # anti_params_with_noise = jax.tree_multimap(
    #     lambda g, n: g - n * perturbation_std, params, noise)
    
    return params_with_noise, noise


@jax.jit
def add_uniform_and_pareto_noise(params, a, scale, key):
    num_vars = len(jax.tree_leaves(params))
    treedef = jax.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    pareto = jax.tree_multimap(
        lambda g, k: jax.random.pareto(k, a, shape=g.shape, dtype=g.dtype), params,
        jax.tree_unflatten(treedef, all_keys))

    uniform = jax.tree_multimap(
        lambda g, k: jax.random.uniform(k, shape=g.shape, dtype=g.dtype, minval=-scale, maxval=scale), params,
        jax.tree_unflatten(treedef, all_keys))

    
    params_with_noise = jax.tree_multimap(lambda g, u, p: g + g*u*p/100.0,
                                          params, uniform, pareto)
    # anti_params_with_noise = jax.tree_multimap(
    #     lambda g, n: g - n * perturbation_std, params, noise)
    
    return params_with_noise, uniform, pareto


@jax.jit
def add_uniform_noise(params, noise_max, key):
    num_vars = len(jax.tree_leaves(params))
    treedef = jax.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.uniform(k, shape=g.shape, dtype=g.dtype, minval=-noise_max, maxval=noise_max), params,
        jax.tree_unflatten(treedef, all_keys))
    
    params_with_noise = jax.tree_multimap(lambda g, n: g + g*n,
                                      params, noise)
    # anti_params_with_noise = jax.tree_multimap(
    #     lambda g, n: g - n * perturbation_std, params, noise)
    
    return params_with_noise, noise

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



@jax.jit
def add_guassian_noise(params, noise_std, key):
    num_vars = len(jax.tree_leaves(params))
    treedef = jax.tree_structure(params)
    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_multimap(
        lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype), params,
        jax.tree_unflatten(treedef, all_keys))
    
    params_with_noise = jax.tree_multimap(lambda g, n: g + n * noise_std,
                                      params, noise)
    anti_params_with_noise = jax.tree_multimap(
        lambda g, n: g - n * noise_std, params, noise)
    
    return params_with_noise, anti_params_with_noise, noise



@partial(jax.jit, static_argnums=(0,1,2,6,7,8,9,10,11,12))
def do_local_apg(num_epochs, env_fn, policy_apply, normalizer_params, policy_params, key, learning_rate=1e-3, episode_length=1000, action_repeat=1, normalize_observations=True, batch_size=1, clipping=1e9, truncation_length=None):

    env = env_fn()
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(policy_params)

    _, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(
          env.observation_size, normalize_observations, num_leading_batch_dims=1)

    clip_init, clip_update = optax.adaptive_grad_clip(clipping, eps=0.001)
    clip_state = clip_init(policy_params)

    def do_training_rollout(policy_params, normalizer_params, key):
        init_state = env.reset(key)
        h0 = jnp.zeros_like(init_state.obs)

        def do_one_rnn_step(carry):
            state, h, policy_params, normalizer_params, reward_sum, step_index  = carry

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
            return (nstate, h1, policy_params, normalizer_params, reward_sum, step_index), None
            #return (nstate, h1, policy_params, normalizer_params, reward_sum), None

        def noop(carry):
            return carry, None

        def body_fn(carry, x):
            done = jnp.any(carry[0].done)
            return jax.lax.cond(done, noop, do_one_rnn_step, carry)
        
        carry , _  = jax.lax.scan(
            body_fn, (init_state, h0, policy_params, normalizer_params, 0.0, 0),
            None,
            length=episode_length // action_repeat)

        reward_sum = carry[-2]
        
        return reward_sum    
    
    # #@jax.jit
    # def sum_rnn_loss(policy_params, normalizer_params, key):
    #     key, reset_key = jax.random.split(key)
    #     reward_sum = do_training_rollout(policy_params, normalizer_params, reset_key)
    #     return -reward_sum

    vmap_rnn_rollout = jax.vmap(do_training_rollout, in_axes=(None, None, 0))

    def sum_rnn_loss(policy_params, normalizer_params, key):
        keys = jax.random.split(key, num=batch_size)
        rewards_sums = vmap_rnn_rollout(policy_params, normalizer_params, keys)
        return -jnp.mean(rewards_sums)

    grad_fn = jax.value_and_grad(sum_rnn_loss)

    def do_one_apg_iter(carry, epoch):
        optimizer_state, clip_state, key, policy_params, normalizer_params = carry
        key, reset_key = jax.random.split(key)
        #reset_key = key
        init_state = env.reset(reset_key)

        rewards, grads = grad_fn(policy_params, normalizer_params, reset_key)
        param_updates, optimizer_state = optimizer.update(grads, optimizer_state)
        param_updates, clip_state = clip_update(param_updates, clip_state, policy_params)
        policy_params = optax.apply_updates(policy_params, param_updates)

        #normalizer_params = obs_normalizer_update_fn(normalizer_params, obs)

        return (optimizer_state, clip_state, key, policy_params, normalizer_params), -rewards
    
    (optimizer_state, clip_state, key, policy_params2, normalizer_params), rewards = jax.lax.scan(do_one_apg_iter, (optimizer_state, clip_state, key, policy_params, normalizer_params), (jnp.array(range(num_epochs))))

    return policy_params2, rewards





@partial(jax.jit, static_argnums=(0,1,2,6,7,8,9,10,11,12))
def do_local_apg_batched(num_epochs, env_fn, policy_apply, normalizer_params, policy_params, key, learning_rate=1e-3, episode_length=1000, action_repeat=1, normalize_observations=True, batch_size=1, clipping=1e9, truncation_length=None):

    env = env_fn()
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(policy_params)

    _, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(
          env.observation_size, normalize_observations, num_leading_batch_dims=1)

    clip_init, clip_update = optax.adaptive_grad_clip(clipping, eps=0.001)
    clip_state = clip_init(policy_params)

    def do_training_rollout(policy_params, normalizer_params, key):
        init_state = env.reset(key)
        h0 = jnp.zeros_like(init_state.obs)

        def do_one_rnn_step(carry):
            state, h, policy_params, normalizer_params, reward_sum, step_index  = carry

            normed_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
            h1 , actions = policy_apply(policy_params, h, normed_obs)
            nstate = env.step(state, actions)
            
            #nstate = jax.lax.cond(True, lambda s,a: env.step(s, a), lambda s,a: s, state,actions)
            #h1 = jax.lax.cond(nstate.done, lambda x: jnp.zeros_like(h1), lambda x: h1, None)
            #jax.lax.stop_gradient(nstate)

            if truncation_length is not None and truncation_length > 0:
                nstate = jax.lax.cond(
                    jnp.mod(step_index + 1, truncation_length) == 0.,
                    jax.lax.stop_gradient, lambda x: x, nstate)


            reward_sum += nstate.reward
            step_index += 1
            return (nstate, h1, policy_params, normalizer_params, reward_sum, step_index), None
            #return (nstate, h1, policy_params, normalizer_params, reward_sum), None

        def noop(carry):
            return carry, None

        def body_fn(carry, x):
            done = jnp.any(carry[0].done)
            return jax.lax.cond(done, noop, do_one_rnn_step, carry)
        
        carry , _  = jax.lax.scan(
            body_fn, (init_state, h0, policy_params, normalizer_params, jnp.zeros(batch_size), jnp.zeros(batch_size)),
            None,
            length=episode_length // action_repeat)

        reward_sum = carry[-2]
        
        return reward_sum    
    
    # #@jax.jit
    # def sum_rnn_loss(policy_params, normalizer_params, key):
    #     key, reset_key = jax.random.split(key)
    #     reward_sum = do_training_rollout(policy_params, normalizer_params, reset_key)
    #     return -reward_sum

    def sum_rnn_loss(policy_params, normalizer_params, key):
        keys = jax.random.split(key, num=batch_size)
        rewards_sums = do_training_rollout(policy_params, normalizer_params, keys)
        return -jnp.mean(rewards_sums)

    grad_fn = jax.value_and_grad(sum_rnn_loss)

    def do_one_apg_iter(carry, epoch):
        optimizer_state, clip_state, key, policy_params, normalizer_params = carry
        key, reset_key = jax.random.split(key)
        #reset_key = key
        init_state = env.reset(reset_key)

        rewards, grads = grad_fn(policy_params, normalizer_params, reset_key)
        param_updates, optimizer_state = optimizer.update(grads, optimizer_state)
        param_updates, clip_state = clip_update(param_updates, clip_state, policy_params)
        policy_params = optax.apply_updates(policy_params, param_updates)

        #normalizer_params = obs_normalizer_update_fn(normalizer_params, obs)

        return (optimizer_state, clip_state, key, policy_params, normalizer_params), -rewards
    
    (optimizer_state, clip_state, key, policy_params2, normalizer_params), rewards = jax.lax.scan(do_one_apg_iter, (optimizer_state, clip_state, key, policy_params, normalizer_params), (jnp.array(range(num_epochs))))

    return policy_params2, rewards



@partial(jax.jit, static_argnums=(0,1,5,6,7))
def do_one_rollout(env_fn, policy_apply, normalizer_params, policy_params, key, episode_length=1000, action_repeat=1, normalize_observations=True):

    env = env_fn()
    _, obs_normalizer_update_fn, obs_normalizer_apply_fn = normalization.create_observation_normalizer(
          env.observation_size, normalize_observations, num_leading_batch_dims=1)

    init_state = env.reset(key)
    h0 = jnp.zeros_like(init_state.obs)

    @jax.jit
    def do_one_rnn_step(carry, step_idx):
        state, h, policy_params, normalizer_params  = carry
        
        normed_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
        h1 , actions = policy_apply(policy_params, h, normed_obs)
        nstate = env.step(state, actions)    
        #h1 = jax.lax.cond(nstate.done, lambda x: jnp.zeros_like(h1), lambda x: h1, None)
        #jax.lax.stop_gradient(nstate)
        return (jax.lax.stop_gradient(nstate), h1, policy_params, normalizer_params), (nstate.reward,state.obs, actions, jax.lax.stop_gradient(nstate))
        #return (nstate, h1, policy_params, normalizer_params), (nstate.reward,state.obs, actions, nstate)

        
    _, (rewards, obs, acts, states) = jax.lax.scan(
        do_one_rnn_step, (init_state, h0, policy_params, normalizer_params),
        (jnp.array(range(episode_length // action_repeat))),
        length=episode_length // action_repeat)
        
    return rewards, obs, acts, states






## basin hopping
