import jax
import jax.numpy as jnp
from flax import linen as nn

class GruController(nn.Module):
    obs_size: int
    act_size: int
    hidden_size:int
    
    @nn.compact
    def __call__(self, carry, x):
        h0 = carry
        y0 = nn.relu(nn.Dense(self.hidden_size)(x))
        h1,y1 = nn.GRUCell()(h0,y0)
        y2 = nn.relu(nn.Dense(self.hidden_size)(y1))
        y3 = nn.relu(nn.Dense(self.hidden_size)(y2))
        a = jnp.tanh(nn.Dense(self.act_size)(y3))
        
        return h1, a


class MlpController(nn.Module):
    obs_size: int
    act_size: int
    hidden_size:int
    
    @nn.compact
    def __call__(self, carry, x):
        y1 = nn.relu(nn.Dense(self.hidden_size)(x))
        y2 = nn.relu(nn.Dense(self.hidden_size)(y1))
        a = jnp.tanh(nn.Dense(self.act_size)(y2))

        return carry, a


class LinearController(nn.Module):
    obs_size: int
    act_size: int
    
    @nn.compact
    def __call__(self, carry, x):
        a = jnp.tanh(nn.Dense(self.act_size)(x))
        return carry, a
