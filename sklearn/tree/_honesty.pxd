from ._tree cimport TreeBuildEvent,
                    TreeBuildEventArgs,
                    TreeBuildEventHandler,
                    TreeBuildEventHandlerEnv,
                    TreeBuildEventHandlerClosure,
                    TreeBuildEventHandlerClosureWrapper


cdef struct HonestTreeBuilderEventHandlerEnv:
    intp_t foo

