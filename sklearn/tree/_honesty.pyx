cdef bint _honest_min_sample_leaf_condition(
    TreeBuildEvent evt,
    BuildEnv* build_env,
    EventHandlerEnv handler_env
    ) noexcept nogil:
    if evt == TreeBuildEvent.ADD_NODE:
        pass

    return True

cdef class HonestMinSampleLeafCondition:
    __cinit__(self, EventHandlerEnv handler_env):
        self.c.f = _honest_min_sample_leaf_condition
        self.c.e = handler_env
