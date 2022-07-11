from function_transformer_attention import ODEFuncTransformerAtt, ExtendedODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc
from function_laplacian_diffusion import ExtendedLaplacianODEFunc
from function_laplacian_diffusion import ExtendedLaplacianODEFunc2
from function_laplacian_diffusion import ExtendedLaplacianODEFunc3
from function_coupled_ode import CoupledODEFunc 
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock
from block_mixed import MixedODEblock
from block_transformer_hard_attention import HardAttODEblock
from block_transformer_rewiring import RewireAttODEblock
from block_coupled_ode import CoupledODEBlock

class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass


def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'mixed':
    block = MixedODEblock
  elif ode_str == 'attention':
    block = AttODEblock
  elif ode_str == 'hard_attention':
    block = HardAttODEblock
  elif ode_str == 'rewire_attention':
    block = RewireAttODEblock
  elif ode_str == 'constant':
    block = ConstantODEblock
  elif ode_str == 'coupled':
    block = CoupledODEBlock
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']

  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'GAT':
    f = ODEFuncAtt
  elif ode_str == 'transformer':
    f = ODEFuncTransformerAtt
  elif ode_str == 'ext_laplacian': # The extended laplacian function
    f = ExtendedLaplacianODEFunc
  elif ode_str == 'ext_laplacian2': # Extended laplacian function 2
    ExtendedLaplacianODEFunc2.alpha_ = opt['alpha_']
    ExtendedLaplacianODEFunc2.clipping_bound = opt['clip_bound']
    f = ExtendedLaplacianODEFunc2
  elif ode_str == 'ext_laplacian3': # Exnteded laplacian function 3 (truncated norm_x)
    ExtendedLaplacianODEFunc3.alpha_ = opt['alpha_']
    ExtendedLaplacianODEFunc3.epsilon_ = opt['epsilon_']
    f = ExtendedLaplacianODEFunc3
  elif ode_str == 'ext_transformer':
    ExtendedODEFuncTransformerAtt.alpha_ = opt['alpha_']
    ExtendedODEFuncTransformerAtt.clipping_bound = opt['clip_bound']
    f = ExtendedODEFuncTransformerAtt
  elif ode_str == 'coupled': # Coupled ODE 
    f = CoupledODEFunc
  else:
    raise FunctionNotDefined
  return f
