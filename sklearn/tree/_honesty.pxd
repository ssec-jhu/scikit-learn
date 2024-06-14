# Authors: Samuel Carliles <scarlil1@jhu.edu>
#
# License: BSD 3 clause

# See _honesty.pyx for details.

from .._splitter cimport Partitioner
from .._tree cimport BuildEnv, EventHandlerEnv, TreeBuildEvent, TreeBuildEventHandler
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint32_t


cdef class BaseHonestEnv:
    cdef:
        const float32_t[:, :] X
        intp_t[::1] samples
        float32_t[::1] feature_values
        Partitioner partitioner

cdef struct Extent:
    intp_t start
    intp_t end

cdef class HonestMinSampleLeafCondition(TreeBuildEventHandler):
    pass
