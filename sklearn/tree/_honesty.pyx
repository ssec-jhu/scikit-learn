from libc.math cimport floor, log2, pow


cdef bint _handle_set_active_parent(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != TreeBuildEvent.SET_ACTIVE_PARENT:
        return True
    
    HonestEnv* env = <HonestEnv*>handler_env
    TreeBuildSetActiveParentEventData* data = <TreeBuildSetActiveParentEventData*>event_data

    if data.parent_node_id < 0 || data.parent_node_id >= env.tree.size():
        return False

    env.active_parent = &(env.tree[data.parent_node_id])

    return True

cdef class SetActiveParentHandler(EventHandler):
    def __cinit__(self, HonestEnv* env):
        self._event_types = [TreeBuildEvent.SET_ACTIVE_PARENT]
        self.event_types = self._event_types

        self.c.f = _handle_set_active_parent
        self.c.e = env


cdef bint _handle_sort_feature(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != NodeSplitEvent.SORT_FEATURE:
        return True
    
    HonestEnv* env = <HonestEnv*>handler_env
    NodeSortFeatureEventData* data = <NodeSortFeatureEventData*>event_data

    env.partitioner.sort_samples_and_feature_values(data.feature)

    return True

cdef class NodeSortFeatureHandler(EventHandler):
    def __cinit__(self, HonestEnv* env):
        self._event_types = [NodeSplitEvent.SORT_FEATURE]
        self.event_types = self._event_types

        self.c.f = _handle_sort_feature
        self.c.e = env


cdef bint _handle_add_node(
    EventType event_type,
    EventHandlerEnv handler_env,
    EventData event_data
) noexcept nogil:
    if event_type != TreeBuildEvent.ADD_NODE:
        return True

    cdef float64_t h, feature_value
    cdef intp_t i, n_left, n_missing, size = env.tree.size()
    cdef HonestEnv* env = <HonestEnv*>handler_env
    cdef TreeBuildAddNodeEventData* data = <TreeBuildAddNodeEventData*>event_data
    cdef Interval *interval, *parent

    if data.node_id >= size:
        # as a heuristic, assume a complete tree and add a level
        h = floor(log2(size))
        env.tree.resize(size + <intp_t>pow(2, h + 1))

    interval = &(env.tree[node_id])

    if data.parent_node_id >= 0:
        parent = &(env.tree[data.parent_node_id])

        # *we* don't need to sort to find the split pos we'll need for partitioning,
        # but the partitioner internals are so stateful we had better just do it
        # to ensure that it's in the expected state
        env.partitioner.init_node_split(parent.low_idx, parent.hi_idx)
        env.partitioner.sort_samples_and_feature_values(parent.feature)

        # count n_left to find split pos
        n_left = 0
        i = parent.low_idx
        feature_value = env.X[env.samples[i], parent.feature]

        while !isnan(feature_value) && feature_value < parent.split_value && i <= parent.hi_idx:
            n_left += 1
            i += 1
            feature_value = env.X[env.samples[i], parent.feature]

        env.partitioner.partition_samples_final(
            parent.low_idx + n_left, parent.split_value, parent.feature, partitioner.n_missing
            )

        if data.is_left:
            interval.low_idx = parent.low_idx
            interval.hi_idx = parent.low_idx + n_left - 1
        else:
            interval.low_idx = parent.low_idx + n_left
            interval.hi_idx = parent.hi_idx
    else:
        # the node being added is the tree root
        interval.low_idx = 0
        interval.hi_idx = env.samples.shape[0] - 1

    interval.feature = data.feature
    interval.split = data.split_value


cdef class AddNodeHandler(EventHandler):
    def __cinit__(self, HonestEnv* env):
        self._event_types = [TreeBuildEvent.ADD_NODE]
        self.event_types = self._event_types

        self.c.f = _handle_add_node
        self.c.e = env

# honest_nodes[stack_record.parent_node_id]:
#  start
#  end
#  feature
#  split_value
#
# stack_record.parent_node_id
# stack_record.is_left
#