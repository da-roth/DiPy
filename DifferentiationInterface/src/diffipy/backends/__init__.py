
from .numpy import *
from .pytorch import *
from .jax import *
from .tensorflow import *


# Try to import Aadc
try:
    import aadc
    AADC_AVAILABLE = True
except ModuleNotFoundError:
    AADC_AVAILABLE = False

# Example usage
if AADC_AVAILABLE:
    from .aadc import *
    #pass
# else:
#     print("aadc module is not available. Some features may be disabled.")

from .BackendHelper import *
