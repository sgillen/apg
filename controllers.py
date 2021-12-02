import jax
import jax.numpy as jnp
from flax import linen as nn

class GruController(nn.Module):
    obs_size: int
    act_size: int
    
    @nn.compact
    def __call__(self, carry, x):
        h0 = carry
        h1,y0 = nn.GRUCell()(h0,x)
        y1 = nn.Dense(64)(y0)
        y2 = nn.Dense(64)(y1)
        a = jnp.tanh(nn.Dense(self.act_size)(y2))


        
        return h1, a
