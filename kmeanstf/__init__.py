import sys
class PythonVersionError(Exception):
    # raised when Python 2 is used
    pass
if sys.version_info[0] < 3:
    raise PythonVersionError("Must be using Python 3 for kmeanstf (version is {:d}.{:d})".format(sys.version_info[0],sys.version_info[1]))
from .kmeanstf import KMeansTF
from .kmeanstf import TunnelKMeansTF
KMeansTF._assert_gpu_memory()